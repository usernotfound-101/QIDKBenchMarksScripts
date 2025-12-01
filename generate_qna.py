#!/usr/bin/env python3
"""
-----------------------------------------------------------------------
  QIDK AI SUSTAINABILITY BENCHMARK SUITE
  Target: Qualcomm Snapdragon (Hexagon NPU via ADB)
-----------------------------------------------------------------------
"""

import os
import json
import time
import random
import subprocess
import sys
from collections import defaultdict

# Importing datasets and handling logging verbosity to keep interface clean
import datasets
datasets.logging.set_verbosity_error()  # This silences the loading bars
from datasets import load_dataset

# ---------------- CONFIGURATION ----------------
# Directories (Managed automatically)
PHONE_TMP_DIR = "/data/local/tmp"
LOCAL_LOG_DIR = "benchmark_logs"
LOCAL_RESULTS_DIR = "benchmark_results"

# Paths on Device
LLAMA_BIN_DIR = os.path.join(PHONE_TMP_DIR, "llama/build-clblast/bin")
LLAMA_CLI_PATH = "./llama-cli"

# Available Models on QIDK
AVAILABLE_MODELS = {

    "1": {"name": "Llama 3.2 (3B)", "file": "Llama-3.2-3B-Instruct-Q5_K_S.gguf"}, 
    "2": {"name": "Phi 3.5 Mini",   "file": "phi-3.5-mini-instruct.Q4_K_M.gguf"},
    "3": {"name": "Gemma 2 (4B)",   "file": "gemma-3-4b-it-q4_0.gguf"},
    "4": {"name": "Qwen 2.5 (4B)",  "file": "qwen3-4b-instruct-2507-q4km.gguf"}
}
# -----------------------------------------------

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    print("\n" + "="*60)
    print("  QIDK EDGE AI BENCHMARKING TOOL  ".center(60))
    print("  Sustainability Assessment Framework  ".center(60))
    print("="*60 + "\n")

def get_user_config():
    """Professional interactive configuration wizard"""
    print_banner()
    
    # 1. Topic Selection
    print(" [1] CONFIGURATION")
    print(" -----------------")
    while True:
        try:
            t_input = input("  > Enter number of topics to cover (e.g., 3): ").strip()
            num_topics = int(t_input)
            if num_topics > 0: break
            print("    [!] Please enter a number greater than 0.")
        except ValueError:
            print("    [!] Invalid input. Please enter a number.")
            
    # 2. Question Count
    while True:
        try:
            q_input = input("  > Enter max questions per topic (e.g., 5): ").strip()
            max_questions = int(q_input)
            if max_questions > 0: break
            print("    [!] Please enter a number greater than 0.")
        except ValueError:
            print("    [!] Invalid input. Please enter a number.")

    # 3. Model Selection Menu
    print("\n [2] SELECT TARGET MODEL")
    print(" -----------------------")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {info['name']}")
    
    selected_model_file = ""
    selected_model_name = ""
    
    while True:
        choice = input("\n  > Select model (1-4): ").strip()
        if choice in AVAILABLE_MODELS:
            selected_model_file = AVAILABLE_MODELS[choice]["file"]
            selected_model_name = AVAILABLE_MODELS[choice]["name"]
            break
        print("    [!] Invalid selection. Please choose 1, 2, 3, or 4.")

    print("\n" + "-"*60)
    print(f"  SUMMARY:")
    print(f"  • Topics: {num_topics}")
    print(f"  • Questions/Topic: {max_questions}")
    print(f"  • Model: {selected_model_name}")
    print("-"*60)
    
    input("\n  Press Enter to start benchmarking...")
    return num_topics, max_questions, selected_model_file

def generate_batches(num_topics, max_questions):
    print("\n  [•] Loading SQuAD dataset (Validation Split)...")
    try:
        # FIXED: Removed quiet=True, using logging.set_verbosity_error() instead
        ds = load_dataset("squad", split="validation")
    except Exception as e:
        print(f"  [!] Error loading dataset: {e}")
        sys.exit(1)

    context_to_questions = defaultdict(list)
    for item in ds:
        context_to_questions[item["context"]].append(item)

    all_topics = list(context_to_questions.keys())
    
    # Randomly select topics
    if num_topics > len(all_topics):
        print(f"  [!] Warning: Requested {num_topics} topics but only {len(all_topics)} available. Using all.")
        num_topics = len(all_topics)
        
    selected_topics = random.sample(all_topics, num_topics)
    
    batches = []
    q_counter = 1
    
    for topic in selected_topics:
        items = context_to_questions[topic][:max_questions]
        batch_data = []
        for item in items:
            qid = f"q{q_counter:04d}"
            q_counter += 1
            batch_data.append({
                "id": qid,
                "question": item["question"].strip(),
                "context": item["context"].strip(),
                "answers": item["answers"]
            })
        batches.append(batch_data)
        
    print(f"  [✓] Prepared {len(batches)} testing batches.")
    return batches

def create_prompt_for_batch(batch_data):
    prompt = """Answer each question using only information from the provided context. Give short, direct answers.

"""
    for item in batch_data:
        prompt += f"[Question {item['id']}]\n"
        prompt += f"Context: {item['context']}\n"
        prompt += f"Question: {item['question']}\n"
        prompt += f"Answer: <your answer here>\n\n"
    return prompt

def push_batch_to_phone(batch_data, batch_idx):
    # Save locally
    batch_file = os.path.join(LOCAL_RESULTS_DIR, f"batch_{batch_idx+1:03d}_data.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    # Create prompt
    prompt = create_prompt_for_batch(batch_data)
    local_prompt_file = os.path.join(LOCAL_RESULTS_DIR, f"batch_{batch_idx+1:03d}_prompt.txt")
    with open(local_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    # Push to phone
    remote_prompt_file = os.path.join(PHONE_TMP_DIR, f"prompts_batch_{batch_idx+1:03d}.txt")
    subprocess.run(["adb", "push", local_prompt_file, remote_prompt_file], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return remote_prompt_file

def run_model_on_phone(remote_prompt_file, batch_idx, model_file):
    remote_result_file = os.path.join(PHONE_TMP_DIR, f"results_batch_{batch_idx+1:03d}.txt")
    model_path = os.path.join(PHONE_TMP_DIR, "gguf", model_file)

    # JSON Schema for structured output
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
    
    adb_cmd = (
        f"cd {LLAMA_BIN_DIR} && "
        f"export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
        f"{LLAMA_CLI_PATH} "
        f"-m {model_path} "
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
    
    start_time = time.time()
    result = subprocess.run(["adb", "shell", adb_cmd], capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n  [!] Error during inference (Batch {batch_idx+1})")
        with open(os.path.join(LOCAL_LOG_DIR, "error_last.log"), "w") as f:
            f.write(result.stderr)
        return None, None, 0

    log_file = os.path.join(LOCAL_LOG_DIR, f"runtime_batch_{batch_idx+1:03d}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(result.stderr)
    
    return remote_result_file, log_file, duration

def pull_results(remote_result_file, batch_idx):
    local_result = os.path.join(LOCAL_RESULTS_DIR, f"results_batch_{batch_idx+1:03d}.json")
    subprocess.run(["adb", "pull", remote_result_file, local_result], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return local_result

def cleanup_remote(files_to_remove):
    cmd = f"rm {' '.join(files_to_remove)}"
    subprocess.run(["adb", "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    # 1. Get Configuration
    num_topics, max_questions, model_file = get_user_config()

    # 2. Setup Directories
    os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)

    # 3. Generate Data
    batches = generate_batches(num_topics, max_questions)
    
    print("\n  [•] Starting Benchmark Execution...\n")
    
    successful_answers = 0
    total_time = 0

    for idx, batch_data in enumerate(batches):
        print(f"  Processing Batch {idx+1}/{len(batches)}...", end="", flush=True)
        
        try:
            # Push
            remote_prompt = push_batch_to_phone(batch_data, idx)
            
            # Run
            remote_result, _, duration = run_model_on_phone(remote_prompt, idx, model_file)
            
            if remote_result:
                # Pull
                pull_results(remote_result, idx)
                
                # Cleanup
                cleanup_remote([remote_prompt, remote_result])
                
                print(f" Done ({duration:.2f}s)")
                total_time += duration
                successful_answers += len(batch_data)
            else:
                print(" Failed.")
                
        except Exception as e:
            print(f" Error: {e}")

    # 4. Final Summary
    print("\n" + "="*60)
    print("  BENCHMARK COMPLETE")
    print("="*60)
    print(f"  • Total Runtime:     {total_time:.2f} seconds")
    print(f"  • Batches Processed: {len(batches)}")
    print(f"  • Est. Latency/Q:    {(total_time/successful_answers if successful_answers else 0):.2f} seconds")
    print(f"  • Logs saved in:     ./{LOCAL_LOG_DIR}")
    print(f"  • Results saved in:  ./{LOCAL_RESULTS_DIR}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()