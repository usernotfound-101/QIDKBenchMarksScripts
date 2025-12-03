#!/usr/bin/env python3
"""
-----------------------------------------------------------------------
  QIDK EDGE AI BENCHMARKING TOOL  (REWORKED)
  - Keeps multi-model selection
  - Stores prompts, results and runtime logs in directories by model name
  - Uses JSON schema for structured model output and robust local parsing
-----------------------------------------------------------------------
"""
import os
import json
import time
import random
import subprocess
import sys
import re
from collections import defaultdict

import datasets
datasets.logging.set_verbosity_error()
from datasets import load_dataset

# ---------------- CONFIGURATION ----------------
PHONE_TMP_DIR = "/data/local/tmp"

# Paths on Device
LLAMA_BIN_DIR = os.path.join(PHONE_TMP_DIR, "llama/build-clblast/bin")
LLAMA_CLI_PATH = "./llama-cli"

# Available Models on QIDK (keep multi-model)
AVAILABLE_MODELS = {
    "1": {"name": "Llama 3.2 (3B)", "file": "Llama-3.2-3B-Instruct-Q5_K_S.gguf", "dir_name": "llama_3.2_3b"},
    "2": {"name": "Phi 3.5 Mini",   "file": "phi-3.5-mini-instruct.Q4_K_M.gguf", "dir_name": "phi_3.5_mini"},
    "3": {"name": "Gemma 2 (4B)",   "file": "gemma-3-4b-it-q4_0.gguf", "dir_name": "gemma_2_4b"},
    "4": {"name": "Qwen 2.5 (4B)",  "file": "qwen3-4b-instruct-2507-q4km.gguf", "dir_name": "qwen_2.5_4b"}
}
# ---------------------------------------------------------------------

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    print("\n" + "="*60)
    print("  QIDK EDGE AI BENCHMARKING TOOL  ".center(60))
    print("  Sustainability Assessment Framework  ".center(60))
    print("="*60 + "\n")

def get_user_config():
    print_banner()
    print(" [1] CONFIGURATION")
    print(" -----------------")
    while True:
        try:
            t_input = input("  > Enter number of topics to cover (e.g., 3): ").strip()
            num_topics = int(t_input)
            if num_topics > 0:
                break
            print("    [!] Please enter a number greater than 0.")
        except ValueError:
            print("    [!] Invalid input. Please enter a number.")
    while True:
        try:
            q_input = input("  > Enter max questions per topic (e.g., 5): ").strip()
            max_questions = int(q_input)
            if max_questions > 0:
                break
            print("    [!] Please enter a number greater than 0.")
        except ValueError:
            print("    [!] Invalid input. Please enter a number.")

    print("\n [2] SELECT TARGET MODEL")
    print(" -----------------------")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {info['name']}")

    selected_model_file = ""
    selected_model_name = ""
    selected_model_dir = ""
    while True:
        choice = input("\n  > Select model (1-4): ").strip()
        if choice in AVAILABLE_MODELS:
            selected_model_file = AVAILABLE_MODELS[choice]["file"]
            selected_model_name = AVAILABLE_MODELS[choice]["name"]
            selected_model_dir = AVAILABLE_MODELS[choice]["dir_name"]
            break
        print("    [!] Invalid selection. Please choose 1, 2, 3, or 4.")

    print("\n" + "-"*60)
    print(f"  SUMMARY:")
    print(f"  • Topics: {num_topics}")
    print(f"  • Questions/Topic: {max_questions}")
    print(f"  • Model: {selected_model_name}")
    print(f"  • Output Directory: {selected_model_dir}")
    print("-"*60)
    input("\n  Press Enter to start benchmarking...")
    return num_topics, max_questions, selected_model_file, selected_model_dir

def generate_batches(num_topics, max_questions):
    print("\n  [•] Loading SQuAD dataset (Validation Split)...")
    try:
        ds = load_dataset("squad", split="validation")
    except Exception as e:
        print(f"  [!] Error loading dataset: {e}")
        sys.exit(1)

    context_to_questions = defaultdict(list)
    for item in ds:
        context_to_questions[item["context"]].append(item)

    all_topics = list(context_to_questions.keys())
    if num_topics > len(all_topics):
        print(f"  [!] Requested {num_topics} topics but only {len(all_topics)} available. Using all.")
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
                "answers": item.get("answers", [])
            })
        batches.append(batch_data)
    print(f"  [✓] Prepared {len(batches)} testing batches.")
    return batches

def create_prompt_for_batch(batch_data):
    prompt = (
        "Answer each question using only information from the provided context.\n"
    )
    for item in batch_data:
        prompt += f"[Question {item['id']}]\n"
        prompt += f"Context: {item['context']}\n"
        prompt += f"Question: {item['question']}\n"
        prompt += "Answer:<your answer here>\n\n"
    return prompt

def push_batch_to_phone(batch_data, batch_idx, model_dir):
    # Create model-specific directories
    local_results_dir = os.path.join("local_results", model_dir)
    local_prompts_dir = os.path.join("local_prompts", model_dir)
    os.makedirs(local_results_dir, exist_ok=True)
    os.makedirs(local_prompts_dir, exist_ok=True)
    
    # Save batch ground truth locally
    batch_file = os.path.join(local_results_dir, f"batch_{batch_idx+1:03d}_data.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)

    # Save prompt locally and push to device
    prompt = create_prompt_for_batch(batch_data)
    local_prompt_file = os.path.join(local_prompts_dir, f"batch_{batch_idx+1:03d}_prompt.txt")
    with open(local_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    remote_prompt_file = os.path.join(PHONE_TMP_DIR, f"prompts_batch_{batch_idx+1:03d}.txt")
    subprocess.run(["adb", "push", local_prompt_file, remote_prompt_file],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return remote_prompt_file

def run_model_on_phone(remote_prompt_file, batch_idx, model_file, model_dir):
    local_log_dir = os.path.join("local_logs", model_dir)
    os.makedirs(local_log_dir, exist_ok=True)
    
    remote_result_file = os.path.join(PHONE_TMP_DIR, f"results_batch_{batch_idx+1:03d}.txt")
    model_path = os.path.join(PHONE_TMP_DIR, "gguf", model_file)

    # JSON Schema for structured output: require "answers"
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
        "required": ["answers"]
    })

    adb_cmd = (
        f"cd {LLAMA_BIN_DIR} && "
        f"export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
        f"{LLAMA_CLI_PATH} "
        f"-m {model_path} "
        f"-c 8192 "
        f"-n 4096 "
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
        err_file = os.path.join(local_log_dir, "error_last.log")
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(result.stderr or "")
        print(f"\n  [!] Error during inference (Batch {batch_idx+1}). See {err_file}")
        return None, None, 0.0

    log_file = os.path.join(local_log_dir, f"runtime_batch_{batch_idx+1:03d}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(result.stderr or "")

    return remote_result_file, log_file, duration

def pull_results(remote_result_file, batch_idx, model_dir):
    local_results_dir = os.path.join("local_results", model_dir)
    os.makedirs(local_results_dir, exist_ok=True)
    
    local_result = os.path.join(local_results_dir, f"results_batch_{batch_idx+1:03d}.json")
    subprocess.run(["adb", "pull", remote_result_file, local_result],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return local_result

def extract_json_from_output(content):
    # try direct parse
    try:
        return json.loads(content)
    except Exception:
        pass
    # find first balanced JSON object
    start = None
    depth = 0
    for i, ch in enumerate(content):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                candidate = content[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    # regex fallback
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def merge_batch_results(json_files, model_dir):
    merged = {"answers": []}
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fin:
                raw = fin.read()
            data = extract_json_from_output(raw)
            if data and isinstance(data.get("answers"), list):
                merged["answers"].extend(data["answers"])
                print(f"  [✓] Merged {len(data['answers'])} answers from {os.path.basename(f)}")
            else:
                print(f"  [⚠] No valid 'answers' array in {os.path.basename(f)}; preview:")
                print(raw[:300].replace("\n", " ") + "...")
        except Exception as e:
            print(f"  [⚠] Could not process {f}: {e}")
    
    local_results_dir = os.path.join("local_results", model_dir)
    merged_file = os.path.join(local_results_dir, "results_all_batches.json")
    with open(merged_file, "w", encoding="utf-8") as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)
    return merged_file

def cleanup_remote(files_to_remove):
    if not files_to_remove:
        return
    cmd = f"rm {' '.join(files_to_remove)}"
    subprocess.run(["adb", "shell", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    random.seed(42)
    num_topics, max_questions, model_file, model_dir = get_user_config()

    batches = generate_batches(num_topics, max_questions)
    pulled_jsons = []
    log_files = []
    remote_files = []

    print("\n  [•] Starting Benchmark Execution...\n")

    for idx, batch_data in enumerate(batches):
        print(f"  Processing Batch {idx+1}/{len(batches)}...", end="", flush=True)
        try:
            remote_prompt = push_batch_to_phone(batch_data, idx, model_dir)
            remote_result, log_file, duration = run_model_on_phone(remote_prompt, idx, model_file, model_dir)
            if remote_result:
                local_json = pull_results(remote_result, idx, model_dir)
                pulled_jsons.append(local_json)
                log_files.append(log_file)
                remote_files.extend([remote_prompt, remote_result])
                print(f" Done ({duration:.2f}s)")
            else:
                print(" Failed.")
        except Exception as e:
            print(f" Error: {e}")

    cleanup_remote(remote_files)
    merged_file = merge_batch_results(pulled_jsons, model_dir)

    print("\n" + "="*60)
    print("  BENCHMARK COMPLETE")
    print("="*60)
    try:
        with open(merged_file, "r", encoding="utf-8") as f:
            merged = json.load(f)
        total_answers = len(merged.get("answers", []))
        print(f"  • Total Answers: {total_answers}")
        if total_answers:
            print("  • Sample:")
            for i, ans in enumerate(merged["answers"][:3]):
                print(f"    {i+1}. id={ans.get('id')} answer={ans.get('answer')}")
    except Exception as e:
        print(f"  [!] Could not read merged results: {e}")

    print(f"\n  • Prompts saved in: {os.path.abspath(os.path.join('local_prompts', model_dir))}")
    print(f"  • Results saved in: {os.path.abspath(os.path.join('local_results', model_dir))}")
    print(f"  • Logs saved in:    {os.path.abspath(os.path.join('local_logs', model_dir))}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()