import json
import re

# Competitor A: Llama
lama_file_path = '/content/drive/MyDrive/Humor_Project/generated_jokes_llama.json'
# Competitor B: Gemma
gemma_file_path = '/content/drive/MyDrive/Humor_Project/outputs_gemmaFinal.jsonl'
# Output File
output_eval_path = '/content/drive/MyDrive/Humor_Project/qwen_evaluation.jsonl'

# 1. SMART DATA LOADER
def load_data_into_dict(file_path):
    dataset = {}
    print(f"📂 Loading: {file_path}...")
    try:
        # Try JSON (Llama Style)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                full_data = json.load(f)
                if "ids" in full_data and isinstance(full_data["ids"], dict):
                    print("   👉 Detected 'Llama-style' nested JSON.")
                    return full_data["ids"]
                if isinstance(full_data, dict):
                    first = next(iter(full_data))
                    if isinstance(full_data[first], dict):
                        print("   👉 Detected Flat JSON Dictionary.")
                        return full_data
            except json.JSONDecodeError: pass

        # Try JSONL (Gemma Style)
        with open(file_path, 'r', encoding='utf-8') as f:
            print("   👉 Detected 'JSONL' structure.")
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if item.get("id"): dataset[item.get("id")] = item
                    except: continue
        print(f"   ✅ Loaded {len(dataset)} items.")
        return dataset
    except FileNotFoundError:
        print(f"   ❌ ERROR: File not found at {file_path}")
        return {}

# 2. Load Data
lama_data = load_data_into_dict(lama_file_path)
gemma_data = load_data_into_dict(gemma_file_path)

# ==========================================
#  PROMPT DEFINITIONS
# ==========================================

def create_user_message(input_type, context, joke_a, joke_b):
    # Logic to handle the specific constraint text
    if input_type == "headline":
        CONSTRAINT_DESCRIPTION = "The joke must satirize or directly relate to the Input Content headline."
        formatted_input = context
    else:
        CONSTRAINT_DESCRIPTION = f"The joke MUST contain the exact words: {context}"
        formatted_input = f"{context}"

    # ADDED: "Write ONLY in English" to prevent Qwen from switching languages
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
**IMPORTANT: Write ONLY in English. Do not generate Chinese.**

{{
  "reasoning": "A single English sentence explaining the winner.",
  "winner": "A" or "B" or "Tie"
}}"""

# 4. Define Parser 
def parse_model_response(response_text):
    #  Clean Markdown
    cleaned = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Try finding the LAST valid JSON object
    try:
        end = cleaned.rfind('}')
        if end != -1:
            start = cleaned.rfind('{', 0, end)
            if start != -1:
                potential_json = cleaned[start:end+1]
                return json.loads(potential_json)
    except: pass

    #  Robust Regex (Use FINDALL to get the LAST match)

    # WINNER Extraction
    winner = "Error"
    winner_matches = re.findall(r'winner["\']?\s*:\s*["\']?(A|B|Tie)["\']?', cleaned, re.IGNORECASE)
    if winner_matches:
        winner = winner_matches[-1].capitalize()

    # REASONING Extraction
    reasoning = "No reasoning found."
    reason_matches = re.findall(r'reasoning["\']?\s*:\s*["\']?(.*?)["\']?,?\s*\n', cleaned, re.IGNORECASE | re.DOTALL)

    if reason_matches:
        reasoning = reason_matches[-1].strip()
    else:
        reasoning = cleaned[-200:].replace('\n', ' ')

    return {"winner": winner, "reasoning": reasoning}

#  Run Loop
stats = {"Llama": 0, "Gemma": 0, "Tie": 0, "Error": 0}
common_ids = sorted(list(set(lama_data.keys()) & set(gemma_data.keys())))

print(f"\n🚀 Starting Evaluation on {len(common_ids)} items...\n")
print(f"{'ID':<10} | {'Winner':<10} | {'Scoreboard'}")
print("-" * 60)

with open(output_eval_path, 'a', encoding='utf-8') as out_f:
    for current_id in common_ids:

        # if current_id < "en_0099" : continue # Testing Logic

        l_entry = lama_data[current_id]
        g_entry = gemma_data[current_id]

        # Inputs
        i_type = l_entry.get("type", "headline")
        i_text = l_entry.get("input_original", l_entry.get("input", ""))
        joke_a = l_entry.get("generated_joke", "")
        joke_b = g_entry.get("generated_joke", "")

        # Generate Prompt
        user_prompt_content = create_user_message(i_type, i_text, joke_a, joke_b)

        # Qwen handles User messages well
        msgs = [{"role": "user", "content": user_prompt_content}]
        inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        # Generate (Temp 0.1 for strict Judging)
        outputs = model.generate(inputs, do_sample=True, max_new_tokens=256, temperature=0.1, top_p=0.9)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean response
        if "OUTPUT FORMAT" in resp:
            resp = resp.split("OUTPUT FORMAT")[-1]

        # Parse
        verdict = parse_model_response(resp)
        w_code = verdict.get("winner", "Error")

        if "A" in w_code: actual = "Llama"; stats["Llama"]+=1
        elif "B" in w_code: actual = "Gemma"; stats["Gemma"]+=1
        elif "Tie" in w_code: actual = "Tie"; stats["Tie"]+=1
        else: actual = "Error"; stats["Error"]+=1

        # Save Result
        rec = {
            "id": current_id,
            "content": i_text,
            "winner": actual,
            "reason": verdict.get("reasoning"),
            "joke_lama": joke_a,
            "joke_gemma": joke_b
        }
        out_f.write(json.dumps(rec) + "\n")

        print(f"{current_id:<10} | {actual:<10} | L:{stats['Llama']} G:{stats['Gemma']} T:{stats['Tie']}")

print("\n🏁 DONE.")

# winner is Llama
# L:726 G:474 T:0