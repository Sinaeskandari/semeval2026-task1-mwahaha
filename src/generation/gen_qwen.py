# installations
# import torch
# major_version, minor_version = torch.cuda.get_device_capability()

# 1. Install build tools FIRST to fix the "subprocess" error
# !pip install --no-deps packaging ninja einops

# # 2. Install Unsloth (latest version)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# # 3. Install dependencies (Removed strict version limits that cause errors)
# !pip install --no-deps xformers trl peft accelerate bitsandbytes



######## downloadig Qwen to google drive

import os
from unsloth import FastLanguageModel
from google.colab import drive

# 1. Mount Drive (Only do this once)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Download QWEN 2.5 from Internet to Colab
print("⬇️ Downloading Qwen 2.5...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # <--- UPDATED to Qwen
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. Save it to your Google Drive PERMANENTLY
save_path = "/content/drive/MyDrive/Humor_Project/Qwen_Model"

print(f"💾 Saving to Drive at: {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Qwen Model successfully saved to Google Drive!")


################  loading model and tokenizer

from unsloth import FastLanguageModel
import torch
from google.colab import drive
import os

# 1. Mount Drive (if not already mounted)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Define path to your SAVED QWEN MODEL
local_model_path = "/content/drive/MyDrive/Humor_Project/Qwen_Model"

print(f"⏳ Loading Qwen from {local_model_path}... please wait...")

# 3. Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,
)
FastLanguageModel.for_inference(model)

print("✅ Success! Qwen is loaded and ready for the generation loop!")


############ generation loop with Qwen

from google.colab import drive
import pandas as pd
import json
import os
import re
from unsloth import FastLanguageModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
output_file = "/content/drive/MyDrive/Humor_Project/outputs_qwenFinal.jsonl"
input_path = '/content/drive/MyDrive/Humor_Project/inputs.tsv'

# Mount Google Drive (if not already mounted)
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Read the file
df = pd.read_csv(input_path, sep='\t')
data = df.to_dict('records')

# ==========================================
# 2. CHECK PROGRESS (Resume capability)
# ==========================================
processed_ids = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                saved_item = json.loads(line)
                processed_ids.add(saved_item['id'])
            except:
                pass
    print(f"↔️ Resuming... Found {len(processed_ids)} jokes already generated.")

# ==========================================
# 3. HELPERS: Safety Filter
# ==========================================
def contains_chinese(text):
    """Returns True if the text contains any Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# ==========================================
# 4. GENERATION LOOP (With Qwen-Specific English Retries)
# ==========================================

print("🚀 Starting Generation with Qwen...")

for row in data:
    current_id = row['id']
    # SKIP if already done
    if current_id in processed_ids:
        continue

    # --- LOGIC: Handle "-" and Empty Values ---
    headline_val = str(row.get('headline', "-")).strip()
    w1_val = str(row.get('word1', "-")).strip()
    w2_val = str(row.get('word2', "-")).strip()

    # --- MATCHING GEMMA'S PROMPT STYLE ---
    if headline_val != "-" and headline_val != "" and headline_val.lower() != "nan":
        # HEADLINE PROMPT
        prompt_text = f"""### Instruction
You are a witty, cynical stand-up comedian. Write ONLY in English.
Your task is to write EXACTLY ONE punchy joke (1-2 sentences) based on the provided headline.

### Examples
Headline: "Study finds 90% of office meetings could be emails."
Joke: "A new study found that 90% of office meetings could be emails, which implies the other 10% could have just been silence."

Headline: "Billionaire builds giant clock inside a mountain."
Joke: "Jeff Bezos is building a giant clock inside a mountain, finally providing a way to tell time for the five people who actually survive the apocalypse."

Headline: "Scientists discover new species of deep-sea jelly."
Joke: "Scientists have discovered a new species of jelly at the bottom of the ocean, mostly because they were tired of looking for the ones in their donuts."

### Task
Headline: "{headline_val}"

### MANDATORY Rules
1. The joke must be STANDALONE. Do not assume the audience has read the headline; include the premise in the joke itself.
2. Be clever, cynical, or ironic.
3. NO explanations or conversational filler (e.g., do not write "Here is the joke"). 
4. Output ONLY the joke.
5. Output MUST be in English. No Chinese characters allowed.

### Response
Joke:"""
        input_type = "headline"
        input_content = headline_val

    else:
        # WORDS PROMPT
        real_w1 = w1_val if w1_val != "-" and w1_val.lower() != "nan" else "something"
        real_w2 = w2_val if w2_val != "-" and w2_val.lower() != "nan" else "random"

        prompt_text = f"""You are a witty, cynical stand-up comedian. You must write in English.
Task: Write EXACTLY ONE punchy joke (1–2 sentences) that connects the following two concepts: "{real_w1}" and "{real_w2}".

Example 1 (Metaphor/Analogy):
Words: "unplug" + "fridge"
Joke: "My current relationship is exactly like an unplugged fridge: it's cold, dark, and I'm terrified to open it and see what's rotting inside."

Example 2 (Ironic Failure):
Words: "hammer" + "banana"
Joke: "I tried to fix my diet with the same tool I use to fix my furniture, but it turns out taking a hammer to a banana just makes a smoothie with too much crunch."

Example 3 (Cynical Observation):
Words: "measure" + "pizza"
Joke: "Trying to measure happiness with money is like trying to measure a pizza with a thermometer: you're using the wrong tool and you're just going to burn your hand."

MANDATORY Rules:
- You can use the words literally OR metaphorically.
- The logic must hold up (e.g., do not say a laptop cooks food).
- Do NOT explain the joke.
- Do NOT use filler like "Here is a joke."
- Output MUST be in English. No Chinese characters allowed.

Words: "{real_w1}", "{real_w2}"
Joke:"""
        input_type = "words"
        input_content = f"{real_w1}, {real_w2}"

    # --- RETRY LOOP FOR QWEN ENGLISH-ONLY CHECK ---
    max_retries = 3
    final_joke = None

    for attempt in range(max_retries):
        # 1. Prepare Inputs (Using only 'user' role for consistency)
        inputs_tf = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        # 2. Generate (Adjusting Temp slightly if retrying)
        
        
        outputs = model.generate(
            inputs_tf,
            do_sample = True,
            max_new_tokens = 128,
            temperature = 0.9,
            top_p = 0.9,
            repetition_penalty = 1.2
        )

        # 3. Parse & Clean
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Isolate the joke part
        if "Joke:" in raw_text:
            temp_joke = raw_text.split("Joke:")[-1]
        else:
            temp_joke = raw_text.split("assistant")[-1] if "assistant" in raw_text else raw_text

        # Cleanup
        temp_joke = temp_joke.replace("assistant\n", "").replace("assistant", "")

        temp_joke = temp_joke.strip()
        temp_joke = temp_joke.split("\n\n")[0]
        temp_joke = temp_joke.strip(' "')

        # 4. English Safety Filter
        if contains_chinese(temp_joke):
            print(f"⚠️ ID {current_id} (Attempt {attempt+1}/{max_retries}): Chinese detected. Retrying...")
            continue
        else:
            final_joke = temp_joke
            break # English joke found, exit retry loop

    # --- FINAL HANDLING ---
    if final_joke is None:
        # Errors are printed, but not saved in a way that breaks the pipeline
        print(f"❌ ID {current_id}: Failed to generate English joke after {max_retries} attempts.")
        # We record it as a failure so we can potentially retry manually later
        final_joke = "ERROR: GENERATION FAILED (CHINESE DETECTED)"

    # Preview
    print(f"ID {current_id}: {final_joke[:50]}...")

    # --- SAVE ---
    result_entry = {
        "id": current_id,
        "type": input_type,
        "input_original": input_content,
        "generated_joke": final_joke
    }

    with open(output_file, "a", encoding='utf-8') as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print("✅ Finished processing all inputs with Qwen!")