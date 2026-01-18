import os
import json
import re
import torch

from unsloth import FastLanguageModel

# ============================================
# PATH CONFIGURATION (COLAB ONLY)
# ============================================
BASE_DIR = '/content/drive/MyDrive/Humor_Project'

llama_rag_path = f"{BASE_DIR}/outputs_Rag_lama.jsonl"
gemma_rag_path = f"{BASE_DIR}/outputs_gemma_rag.jsonl"
qwen_rag_path = f"{BASE_DIR}/outputs_qwen_rag.jsonl"
output_eval_path = f"{BASE_DIR}/llama_evaluation.jsonl"

def load_data_into_dict(file_path):
    dataset = {}
    print(f"Loading: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                full_data = json.load(f)
                if "ids" in full_data and isinstance(full_data["ids"], dict):
                    return full_data["ids"]
                if isinstance(full_data, dict):
                    first = next(iter(full_data))
                    if isinstance(full_data[first], dict):
                        return full_data
            except json.JSONDecodeError:
                pass

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("id"):
                            dataset[item.get("id")] = item
                    except:
                        continue
        print(f"Loaded {len(dataset)} items.")
        return dataset
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return {}

llama_data = load_data_into_dict(llama_rag_path)
gemma_data = load_data_into_dict(gemma_rag_path)
qwen_data = load_data_into_dict(qwen_rag_path)

def create_user_message(input_type, context, joke_a, joke_b):
    if input_type == "headline":
        CONSTRAINT_DESCRIPTION = "The joke must satirize or directly relate to the Input Content headline."
        formatted_input = context
    else:
        CONSTRAINT_DESCRIPTION = f"The joke MUST contain the exact words: {context}"
        formatted_input = f"{context}"

    return f"""You are an impartial humor judge. You evaluate AI-generated jokes based on quality and adherence to specific constraints.
I will provide you with a context and two jokes (Joke A and Joke B).
Your goal is to choose the winner based on two criteria:
1. CONSTRAINT CHECK: Does the joke follow the specific rule provided below?
2. QUALITY: Which joke is funnier, more clever, or better written?

### CONTEXT
Constraint Rule: {CONSTRAINT_DESCRIPTION}
Input Content: {formatted_input}

### THE CANDIDATES
Joke A (Llama):
{joke_a}

Joke B (Gemma):
{joke_b}

### EVALUATION STEPS
1. Analyze if Joke A satisfies the Constraint Rule.
2. Analyze if Joke B satisfies the Constraint Rule.
3. Compare the humor and wit of both jokes.
4. If one joke follows the rule and the other does not, the one that follows the rule MUST win.
5. If both follow the rule (or both fail), pick the funniest one.

### OUTPUT FORMAT
Provide your final decision in the following JSON format. Do not output any other text or markdown.
Keeping the reasoning concise (MAXIMUM 1 SENTENCE).

{{
  "reasoning": "A single English sentence explaining the winner.",
  "winner": "A" or "B" or "Tie"
}}"""

def parse_model_response(response_text):
    cleaned = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    try:
        end = cleaned.rfind('}')
        if end != -1:
            start = cleaned.rfind('{', 0, end)
            if start != -1:
                potential_json = cleaned[start:end+1]
                return json.loads(potential_json)
    except:
        pass

    winner = "Error"
    winner_matches = re.findall(r'winner["\']?\s*:\s*["\']?(A|B|Tie)["\']?', cleaned, re.IGNORECASE)
    if winner_matches:
        winner = winner_matches[-1].capitalize()

    reasoning = "No reasoning found."
    reason_matches = re.findall(r'reasoning["\']?\s*:\s*["\']?(.*?)["\']?,?\s*\n', cleaned, re.IGNORECASE | re.DOTALL)

    if reason_matches:
        reasoning = reason_matches[-1].strip()
    else:
        reasoning = cleaned[-200:].replace('\n', ' ')

    return {"winner": winner, "reasoning": reasoning}

try:
    torch.cuda.empty_cache()
    
    print("Loading Llama with Unsloth optimization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    print("Model loaded with Unsloth optimization!\n")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have: pip install -U bitsandbytes transformers torch unsloth")
    exit(1)

stats = {"Gemma": 0, "Qwen": 0, "Tie": 0, "Error": 0}
common_ids = sorted(list(set(gemma_data.keys()) & set(qwen_data.keys())))

print(f"\nStarting Evaluation on {len(common_ids)} items...\n")
print(f"{'ID':<10} | {'Winner':<10} | {'Scoreboard'}")
print("-" * 60)

with open(output_eval_path, 'w', encoding='utf-8') as out_f:
    for idx, current_id in enumerate(common_ids):
        g_entry = gemma_data[current_id]
        q_entry = qwen_data[current_id]

        i_type = g_entry.get("type", "headline")
        i_text = g_entry.get("input_original", g_entry.get("input", ""))
        joke_a = g_entry.get("generated_joke", "")
        joke_b = q_entry.get("generated_joke", "")

        if not joke_a or not joke_b:
            continue

        user_prompt_content = create_user_message(i_type, i_text, joke_a, joke_b)

        msgs = [{"role": "user", "content": user_prompt_content}]
        inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                do_sample=True,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9
            )

        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "OUTPUT FORMAT" in resp:
            resp = resp.split("OUTPUT FORMAT")[-1]

        verdict = parse_model_response(resp)
        w_code = verdict.get("winner", "Error")

        if "A" in w_code:
            actual = "Gemma"
            stats["Gemma"] += 1
        elif "B" in w_code:
            actual = "Qwen"
            stats["Qwen"] += 1
        elif "Tie" in w_code:
            actual = "Tie"
            stats["Tie"] += 1
        else:
            actual = "Error"
            stats["Error"] += 1

        rec = {
            "id": current_id,
            "content": i_text,
            "winner": actual,
            "reason": verdict.get("reasoning"),
            "joke_gemma": joke_a,
            "joke_qwen": joke_b
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0:
            print(f"{current_id:<10} | {actual:<10} | G:{stats['Gemma']} Q:{stats['Qwen']} T:{stats['Tie']}")

print("\nFinal Results:")
print(f"Gemma wins: {stats['Gemma']}")
print(f"Qwen wins: {stats['Qwen']}")
print(f"Ties: {stats['Tie']}")
print(f"Errors: {stats['Error']}")

# Determine overall winner and save summary
overall_winner = "Gemma" if stats['Gemma'] > stats['Qwen'] else ("Qwen" if stats['Qwen'] > stats['Gemma'] else "Tie")
summary = {
    "judge_model": "Meta-Llama-3.1-8B-Instruct",
    "comparison": "Gemma_RAG vs Qwen_RAG",
    "gemma_wins": stats['Gemma'],
    "qwen_wins": stats['Qwen'],
    "ties": stats['Tie'],
    "errors": stats['Error'],
    "total_evaluated": stats['Gemma'] + stats['Qwen'] + stats['Tie'] + stats['Error'],
    "overall_winner": overall_winner
}

summary_file = f'{BASE_DIR}/summary_llama_judge.json'
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n📊 Summary saved to: {summary_file}")
print(f"🏆 Overall Winner (Llama Judge): {overall_winner}")
