#!/usr/bin/env python3
"""
squad_benchmark_adb_batches.py

Automates SQuAD batch processing on Android via ADB:
1. Generate question batches from SQuAD
2. Create prompts that ask the model to answer questions
3. Push each batch to the phone
4. Run llama-cli per batch with JSON output
5. Pull all batch results and merge locally
"""

import os
import json
import random
import subprocess
from datasets import load_dataset
from collections import defaultdict

# ---------------- CONFIG ----------------
PHONE_TMP_DIR = "/data/local/tmp"
MODEL_PATH = os.path.join(PHONE_TMP_DIR, "gguf/phi-3.5-mini-instruct.Q4_K_M.gguf")
LLAMA_BIN_DIR = os.path.join(PHONE_TMP_DIR, "llama/build-clblast/bin")
LLAMA_CLI_PATH = "./llama-cli"
LOG_FILE = os.path.join(PHONE_TMP_DIR, "logs.txt")
BATCH_SIZE = 10
NUM_BATCHES = 3
# ----------------------------------------

def generate_batches(num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    print("Loading SQuAD validation split...")
    ds = load_dataset("squad", split="validation")

    context_to_questions = defaultdict(list)
    for item in ds:
        context_to_questions[item["context"]].append(item)

    topics = list(context_to_questions.keys())

    batches = []
    q_counter = 1
    for batch_idx in range(num_batches):
        topic = random.choice(topics)
        items = random.sample(context_to_questions[topic], min(batch_size, len(context_to_questions[topic])))
        batch_data = []
        for item in items:
            qid = f"q{q_counter:04d}"
            q_counter += 1
            batch_data.append({
                "id": qid,
                "question": item["question"].strip(),
                "context": item["context"].strip(),
                "answers": item["answers"]  # Ground truth for later evaluation
            })
        batches.append(batch_data)
    print(f"✓ Generated {num_batches} batches with {batch_size} questions each")
    return batches

def create_prompt_for_batch(batch_data):
    """Create a prompt that asks the model to answer questions based on context"""
    prompt = """Answer each question using only information from the provided context. Give short, direct answers.

"""
    
    for item in batch_data:
        prompt += f"[Question {item['id']}]\n"
        prompt += f"Context: {item['context']}\n"
        prompt += f"Question: {item['question']}\n"
        prompt += f"Answer: <your answer here>\n\n"
    
    return prompt

def push_batch_to_phone(batch_data, batch_idx):
    # Save the batch data with ground truth for later evaluation
    batch_file = f"batch_{batch_idx+1:03d}_data.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    # Create the prompt file
    prompt = create_prompt_for_batch(batch_data)
    local_prompt_file = f"batch_{batch_idx+1:03d}_prompt.txt"
    with open(local_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    remote_prompt_file = os.path.join(PHONE_TMP_DIR, f"prompts_batch_{batch_idx+1:03d}.txt")
    subprocess.run(["adb", "push", local_prompt_file, remote_prompt_file], check=True)
    print(f"✓ Pushed batch {batch_idx+1} to phone at {remote_prompt_file}")
    return remote_prompt_file

def run_llama_batch_on_phone(remote_prompt_file, batch_idx):
    remote_result_file = os.path.join(PHONE_TMP_DIR, f"results_batch_{batch_idx+1:03d}.txt")
    
    json_schema = json.dumps({
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "answer": {"type": "string"}
                    },
                    "required": ["id", "answer"]
                }
            }
        },
        "required": ["questions"]
    })
    
    # Simple redirection - stdout (model output) goes to file, stderr (logs) stays on terminal
    adb_cmd = (
        f"cd {LLAMA_BIN_DIR} && "
        f"export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
        f"{LLAMA_CLI_PATH} "
        f"-m {MODEL_PATH} "
        f"-c 8192 "
        f"-n 2048 "
        f"-t 4 "
        f"-no-cnv "
        f"--temp 0.1 "
        f"--top-p 0.9 "
        f"--json-schema '{json_schema}' "
        f"--file {remote_prompt_file} "
        f"> {remote_result_file}"
    )
    
    print(f"Running llama-cli for batch {batch_idx+1}...")
    result = subprocess.run(["adb", "shell", adb_cmd], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running llama-cli:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, adb_cmd)
    
    # Save the runtime logs (stderr) to a local file
    log_file = f"runtime_batch_{batch_idx+1:03d}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(result.stderr)
    
    print(f"✓ Ran llama-cli for batch {batch_idx+1}")
    print(f"  - Model output: {remote_result_file}")
    print(f"  - Runtime logs: {log_file}")
    
    return remote_result_file, log_file

def pull_batch_results(remote_result_file, batch_idx):
    # Pull just the model output (clean JSON)
    local_result = f"results_batch_{batch_idx+1:03d}.json"
    subprocess.run(["adb", "pull", remote_result_file, local_result], check=True)
    
    print(f"✓ Pulled batch {batch_idx+1} results to {local_result}")
    
    return local_result

def extract_json_from_output(content):
    """Extract JSON from model output that may contain other text like the prompt"""
    import re
    
    # Look for JSON object pattern
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return None
    return None

def merge_batch_results(json_files):
    """Merge all JSON result files"""
    merged = {"answers": []}
    
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fin:
                content = fin.read()
            
            # Extract JSON (in case there's prompt echo before it)
            data = extract_json_from_output(content)
            
            if data and "answers" in data:
                merged["answers"].extend(data["answers"])
                print(f"✓ Merged {len(data['answers'])} answers from {f}")
            else:
                print(f"⚠ Warning: No valid JSON found in {f}")
                print(f"  Content preview: {content[:300]}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not process {f}: {e}")
    
    merged_file = "results_all_batches.json"
    with open(merged_file, "w", encoding="utf-8") as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)
    print(f"\n✓ Merged all batch results into {merged_file}")
    return merged_file

def main():
    batches = generate_batches()
    json_files = []
    log_files = []

    for idx, batch_data in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"Processing batch {idx+1}/{len(batches)}")
        print(f"{'='*60}")
        
        # Push prompt to phone
        remote_prompt = push_batch_to_phone(batch_data, idx)
        
        # Run llama-cli on phone (captures logs to local file)
        remote_result, log_file = run_llama_batch_on_phone(remote_prompt, idx)
        
        # Pull clean JSON output from phone
        local_json = pull_batch_results(remote_result, idx)
        
        json_files.append(local_json)
        log_files.append(log_file)

    # Merge all JSON results
    merged_file = merge_batch_results(json_files)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    try:
        with open(merged_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["answers"]:
            print(f"\n✓ Total answers generated: {len(data['answers'])}")
            print(f"\nSample answers (first 3):")
            for i, answer in enumerate(data["answers"][:3]):
                print(f"\n{i+1}. ID: {answer['id']}")
                print(f"   Answer: {answer['answer']}")
            print(f"\n{'='*60}")
            print(f"📄 Results: {merged_file}")
            print(f"📊 Runtime logs: {', '.join(log_files)}")
        else:
            print("\n⚠ Warning: No answers were generated!")
            print("Check the log files for errors.")
    except Exception as e:
        print(f"\n⚠ Error reading merged results: {e}")

if __name__ == "__main__":
    main()