#downloading the model into the google drive

import pandas as pd
import os
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd


from unsloth import FastLanguageModel
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Download model from Internet to Colab first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. Save it to your Google Drive PERMANENTLY
# This might take 2-5 minutes to copy
save_path = "/content/drive/MyDrive/Humor_Project/Gemma_Model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model saved to Google Drive at: {save_path}")
########## downloading and saving Gemma model to Google Drive

from google.colab import drive
drive.mount('/content/drive')

from unsloth import FastLanguageModel
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Download model from Internet to Colab first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 3. Save it to your Google Drive PERMANENTLY
# This might take 2-5 minutes to copy
save_path = "/content/drive/MyDrive/Humor_Project/Gemma_Model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model saved to Google Drive at: {save_path}")




# running the model

from unsloth import FastLanguageModel
import torch
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Define path to your saved model
local_model_path = "/content/drive/MyDrive/Humor_Project/Gemma_Model"

print("⏳ Loading Model & Tokenizer... please wait...")

# 3. THIS is where 'tokenizer' is defined!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,
)
FastLanguageModel.for_inference(model)

print("✅ Success! Now 'tokenizer' is defined. Go run your loop!")




######### generation loop with Gemma
from google.colab import drive
import pandas as pd
import json
import os
from unsloth import FastLanguageModel

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
output_file = "/content/drive/MyDrive/Humor_Project/outputs_gemmaFinal.jsonl"
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
    with open(output_file, 'r') as f:
        for line in f:
            try:
                saved_item = json.loads(line)
                processed_ids.add(saved_item['id'])
            except:
                pass
    print(f"↔️ Resuming... Found {len(processed_ids)} jokes already generated.")

# ==========================================
# 3. GENERATION LOOP (Static Prompts)
# ==========================================

print("🚀 Starting Generation with Gemma...")

for row in data:
    current_id = row['id']

    # if current_id < "en_0080": continue ######### added for testing

    # SKIP if already done
    if current_id in processed_ids:
        continue

    # --- LOGIC: Handle "-" and Empty Values ---
    headline_val = str(row.get('headline', "-")).strip()
    w1_val = str(row.get('word1', "-")).strip()
    w2_val = str(row.get('word2', "-")).strip()

    # --- STATIC PROMPTS ---
    # No more 'previous_joke_structure' variable.
    # We added the static rule: "- Do NOT use repetitive sentence structures."

    if headline_val != "-" and headline_val != "" and headline_val.lower() != "nan":
        # HEADLINE PROMPT
        prompt_text = f"""### Instruction
You are a witty, cynical stand-up comedian.
Your task is to write EXACTLY ONE punchy joke (1-2 sentences) based on the provided headline.

### Examples
Here is how to turn a headline into a standalone joke (weaving the context into the setup):

Headline: "Study finds 90% of office meetings could be emails."
Joke: "A new study found that 90% of office meetings could be emails, which implies the other 10% could have just been silence."

Headline: "Billionaire builds giant clock inside a mountain."
Joke: "Jeff Bezos is building a giant clock inside a mountain, finally providing a way to tell time for the five people who actually survive the apocalypse."

Headline: "Scientists discover new species of deep-sea jelly."
Joke: "Scientists have discovered a new species of jelly at the bottom of the ocean, mostly because they were tired of looking for the ones in their donuts."

### Task
Headline: "{headline_val}"

### Constraints
1. The joke must be STANDALONE. Do not assume the audience has read the headline; include the premise in the joke itself.
2. Be clever, cynical, or ironic.
3. NO explanations or conversational filler (e.g., do not write "Here is the joke").
4. Output ONLY the joke.

### Response
Joke:"""
        input_type = "headline"
        input_content = headline_val

    else:
        # WORDS PROMPT
        real_w1 = w1_val if w1_val != "-" and w1_val.lower() != "nan" else "something"
        real_w2 = w2_val if w2_val != "-" and w2_val.lower() != "nan" else "random"

        prompt_text = f"""You are a witty, cynical stand-up comedian.

Task: Write EXACTLY ONE punchy joke (1–2 sentences) that connects the following two concepts: "{real_w1}" and "{real_w2}".

Here are examples of how to connect random words creatively:

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

Words: "{real_w1}", "{real_w2}"
Joke:"""
        input_type = "words"
        input_content = f"{real_w1}, {real_w2}"

    # --- PREPARE INPUTS ---
    inputs_tf = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")

    # --- GENERATE (Corrected Sampling) ---
    outputs = model.generate(
        inputs_tf,
        do_sample = True,
        max_new_tokens = 128,
        temperature = 0.9,
        top_p = 0.9,
        repetition_penalty = 1.2
    )

    # --- PARSE & CLEAN OUTPUT ---
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 1. Isolate the answer
    if "Joke:" in raw_text:
        final_joke = raw_text.split("Joke:")[-1]
    else:
        # Fallback for Gemma
        final_joke = raw_text.split("model")[-1]

    # 2. CLEANUP 💣
    final_joke = final_joke.replace("model", "") # Keeping this for Gemma
    final_joke = final_joke.strip()
    final_joke = final_joke.split("\n\n")[0]
    final_joke = final_joke.strip(' "')

    # Print a preview
    print(f"ID {current_id}: {final_joke[:50]}...")

    # --- SAVE ---
    result_entry = {
        "id": current_id,
        "type": input_type,
        "input_original": input_content,
        "generated_joke": final_joke
    }

    # Save instantly to JSONL
    with open(output_file, "a", encoding='utf-8') as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print("✅ Finished processing all inputs!")





#testing model parameters
# #gemma test 1
# temp  = 1.0
# min_p = 0.1
# repetition_penalty = 1.2
# #gemma test 2
# temp  = 0.9
# top_p = 0.9
# repetition_penalty = 1.1
# max_new_tokens = 128
# #winner test 2
# #########################################
# #test 3
# temperature = 0.9,
# min_p = 0.4,
# repetition_penalty = 1.1
# #winner test 2
# ##########################################
# #test 4
# temperature = 0.9,
# top_p = 0.8,
# repetition_penalty = 1.1
# #winner test 2
# #########################################
# #gemma test 5
# temp  = 1.1
# top_p = 0.9
# repetition_penalty = 1.1
# #winner test 2
# #########################################
# #test 6
# temperature = 1.1,
# top_p = 0.9,
# repetition_penalty = 1.1
# max_new_tokens = 64

# #winner test 2
# #########################################
# #test 7
# temperature = 1.1,
# top_p = 0.9,
# repetition_penalty = 1.1
# max_new_tokens = 90
# #winner test 2
# #########################################
# #test 8
# temperature = 1.1,
# top_p = 0.9,
# repetition_penalty = 1.1
# max_new_tokens = 150,
# #winner test 2
# #########################################
# #test 9
# temperature = 0.9,
# top_p = 0.9,
# repetition_penalty = 1.1
# max_new_tokens = 128,
# #winner test 2
# #########################################
# #test 10
# temperature = 1.2,
# top_p = 0.9,
# repetition_penalty = 1.1
# max_new_tokens = 128,
# #winner test 2
# #########################################
# #test 11
# temperature = 1.1,
# top_p = 0.9,
# repetition_penalty = 1.2
# max_new_tokens = 128,
# #winner test 11.   wowwwwwwww
# #########################################
# #test 12
# temperature = 1.1,
# min_p = 0.2,
# repetition_penalty = 1.2
# max_new_tokens = 128,
# #winner test 11
# #########################################
# #test 13
# temperature = 1.1,
# min_p = 0.3,
# repetition_penalty = 1.2
# max_new_tokens = 128,
# #winner test 11
# #########################################
