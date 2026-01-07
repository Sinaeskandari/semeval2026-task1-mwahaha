import json
import re

lama_file_path = '/content/drive/MyDrive/Humor_Project/generated_jokes_llama.json' 
qwen_file_path = '/content/drive/MyDrive/Humor_Project/outputs_gemmaFinal.jsonl' 
output_eval_path = '/content/drive/MyDrive/Humor_Project/gemma_evaluation.jsonl'

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

        # Try JSONL (Qwen/Gemma Style)
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
qwen_data = load_data_into_dict(qwen_file_path)

# ==========================================
# 3. PROMPT DEFINITIONS (UPDATED)
# ==========================================

# Define the System Message Constant

def create_user_message(input_type, context, joke_a, joke_b):
    # Logic to handle the specific constraint text
    if input_type == "headline":
        CONSTRAINT_DESCRIPTION = "The joke must satirize or directly relate to the Input Content headline."
        formatted_input = context
    else:
        CONSTRAINT_DESCRIPTION = f"The joke MUST contain the exact words: {context}"
        formatted_input = f"{context}"

    # Return the formatted string EXACTLY as you requested
    return f"""You are an impartial humor judge. You evaluate AI-generated jokes based on quality and adherence to specific constraints.
I will provide you with a context and two jokes (Joke A and Joke B).
Your goal is to choose the winner based on two criteria:
1. CONSTRAINT CHECK: Does the joke follow the specific rule provided below?
2. QUALITY: Which joke is funnier, more clever, or better written?

### CONTEXT
Constraint Rule: {CONSTRAINT_DESCRIPTION}
Input Content: {formatted_input}

### THE CANDIDATES
Joke A:
{joke_a}

Joke B:
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
  "reasoning": "Explain your logic here. Mention if constraints were met.",
  "winner": "A" or "B" or "Tie"
}}"""

# 4. Define Parser
def parse_model_response(response_text):
    # 1. Clean Markdown
    cleaned = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # 2. Try finding the LAST valid JSON object
    # We look for the last occurrence of '}' and the corresponding '{'
    try:
        end = cleaned.rfind('}')
        if end != -1:
            # excessive simple search backwards for start
            start = cleaned.rfind('{', 0, end)
            if start != -1:
                potential_json = cleaned[start:end+1]
                return json.loads(potential_json)
    except: pass

    # 3. Robust Regex (Use FINDALL to get the LAST match)
    
    # WINNER Extraction
    winner = "Error"
    # Find ALL matches of winner property
    winner_matches = re.findall(r'winner["\']?\s*:\s*["\']?(A|B|Tie)["\']?', cleaned, re.IGNORECASE)
    if winner_matches:
        # Take the LAST one [-1] because that's what the model generated
        winner = winner_matches[-1].capitalize()
    
    # REASONING Extraction
    reasoning = "No reasoning found."
    # Find ALL matches of reasoning property
    # The regex matches: reasoning : "..." 
    reason_matches = re.findall(r'reasoning["\']?\s*:\s*["\']?(.*?)["\']?,?\s*\n', cleaned, re.IGNORECASE | re.DOTALL)
    
    if reason_matches:
        # Take the LAST one [-1] to avoid the placeholder text
        reasoning = reason_matches[-1].strip()
    else:
        # Fallback: if strictly parsing keys fails, try to grab the last chunk of text
        reasoning = cleaned[-200:].replace('\n', ' ')

    return {"winner": winner, "reasoning": reasoning}

# 5. Run Loop
stats = {"Llama": 0, "Qwen": 0, "Tie": 0, "Error": 0}
common_ids = sorted(list(set(lama_data.keys()) & set(qwen_data.keys())))

print(f"\n🚀 Starting Evaluation on {len(common_ids)} items...\n")
print(f"{'ID':<10} | {'Winner':<10} | {'Scoreboard'}")
print("-" * 60)

with open(output_eval_path, 'a', encoding='utf-8') as out_f:
    for current_id in common_ids:

        # if current_id < "en_0099" : continue ##########testing reason

        l_entry = lama_data[current_id]
        q_entry = qwen_data[current_id]
        
        # Inputs
        i_type = l_entry.get("type", "headline")
        i_text = l_entry.get("input_original", l_entry.get("input", ""))
        joke_a = l_entry.get("generated_joke", "")
        joke_b = q_entry.get("generated_joke", "")
        
        # --- CHANGES START HERE ---
        
        # 1. Generate the User Prompt String
        user_prompt_content = create_user_message(i_type, i_text, joke_a, joke_b)
        
        # 2. Create the Messages List with SYSTEM ROLE
        msgs = [
            {"role": "user", "content": user_prompt_content}
        ]
        
        # 3. Apply Template (Tokenizer handles the system role formatting)
        inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        # --- CHANGES END HERE ---
        
        outputs = model.generate(inputs, do_sample=True, max_new_tokens=256, temperature=0.1, top_p=0.9)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean response if it repeats the prompt (common in some templates)
        if "OUTPUT FORMAT" in resp: 
            resp = resp.split("OUTPUT FORMAT")[-1]
        
        # Save
        verdict = parse_model_response(resp)
        w_code = verdict.get("winner", "Error")
        
        if w_code == "A": actual = "Llama"; stats["Llama"]+=1
        elif w_code == "B": actual = "Qwen"; stats["Qwen"]+=1
        elif w_code == "Tie": actual = "Tie"; stats["Tie"]+=1
        else: actual = "Error"; stats["Error"]+=1
        
        rec = {"id": current_id,"content": i_text,"winner": actual, "reason": verdict.get("reasoning"), "joke_lama": joke_a, "joke_qwen": joke_b}
        out_f.write(json.dumps(rec) + "\n")
        
        print(f"{current_id:<10} | {actual:<10} | L:{stats['Llama']} Q:{stats['Qwen']} T:{stats['Tie']}")

print("DONE.")