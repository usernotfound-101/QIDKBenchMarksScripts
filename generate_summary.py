#!/usr/bin/env python3
"""
-----------------------------------------------------------------------
  QIDK AI SUSTAINABILITY BENCHMARK SUITE
  Task: Text Summarization (Direct ADB Implementation)
  
  Fixes:
  - Bypasses broken crun-cli.sh script
  - Uses known working paths from generate_qna.py
  - Saves prompt to file to avoid command-line length limits
-----------------------------------------------------------------------
"""

import re
import json
import subprocess
import sys
import os
import time

# ---------------- CONFIGURATION ----------------
# Directories
PHONE_TMP_DIR = "/data/local/tmp"
LOCAL_TEMP_DIR = "_temp_summary_workspace"  # Temp folder for prompt files

# Paths on Device (MATCHING YOUR WORKING QnA SCRIPT)
LLAMA_BIN_DIR = os.path.join(PHONE_TMP_DIR, "llama/build-clblast/bin")
LLAMA_CLI_PATH = "./llama-cli"

# Available Models
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
    print("  Task: Article Summarization  ".center(60))
    print("="*60 + "\n")

def get_user_config():
    print_banner()
    
    # 1. File Inputs
    print(" [1] FILE CONFIGURATION")
    
    # Articles File
    while True:
        f_input = input("  > Enter articles filename (Default 'articles.txt'): ").strip()
        articles_file = f_input if f_input else "articles.txt"
        if os.path.exists(articles_file):
            print(f"    [✓] Found: {articles_file}")
            break
        print(f"    [!] File '{articles_file}' not found.")

    # Highlights File
    while True:
        h_input = input("  > Enter highlights filename (Default 'highlights.txt'): ").strip()
        highlights_file = h_input if h_input else "highlights.txt"
        if os.path.exists(highlights_file):
            print(f"    [✓] Found: {highlights_file}")
            break
        print(f"    [!] File '{highlights_file}' not found.")

    # Output File
    out_input = input("  > Enter output JSON name (Default 'summaries_output.json'): ").strip()
    output_json = out_input if out_input else "summaries_output.json"

    # 2. Model Selection
    print("\n [2] SELECT TARGET MODEL")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {info['name']}")
    
    selected_model_file = ""
    while True:
        choice = input("\n  > Select model (Default 1): ").strip()
        if not choice: choice = "1"
        if choice in AVAILABLE_MODELS:
            selected_model_file = AVAILABLE_MODELS[choice]["file"]
            break
        print("    [!] Invalid selection.")

    input("\n  Press Enter to start processing...")
    return articles_file, highlights_file, output_json, selected_model_file

def read_file_sections(filepath, delimiter_pattern):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []
    
    sections = re.split(delimiter_pattern + r'\s*\n', content)
    if sections[0].strip() == '': sections = sections[1:]
    
    result = []
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            result.append((sections[i], sections[i + 1].strip()))
    return result

def push_prompt_to_phone(article_text, idx):
    """Saves the prompt to a file and pushes it to the phone"""
    # Create the prompt text
    prompt_text = (
        f"{{task begin}}Summarize this article briefly in 3-5 sentences, "
        f"like 'X did this, Y did that', and wrap your summary with "
        f"[summary begin] and [summary end] tags{{task end}}:\n\n{article_text}"
    )
    
    # Save locally
    local_file = os.path.join(LOCAL_TEMP_DIR, f"prompt_{idx}.txt")
    with open(local_file, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    # Push to phone
    remote_file = os.path.join(PHONE_TMP_DIR, f"summary_prompt_{idx}.txt")
    subprocess.run(["adb", "push", local_file, remote_file], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return remote_file

def run_inference_on_phone(remote_prompt_file, model_file):
    model_path = os.path.join(PHONE_TMP_DIR, "gguf", model_file)
    
    # Run the command directly (Standard Llama-CLI arguments)
    # -n 400: Limit output to 400 tokens
    # --no-cnv: No conversation prompt formatting (raw completion)
    adb_cmd = (
        f"cd {LLAMA_BIN_DIR} && "
        f"export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
        f"{LLAMA_CLI_PATH} "
        f"-m {model_path} "
        f"-c 8192 " 
        f"-n 400 "
        f"-t 4 "
        f"-no-cnv "
        f"--file {remote_prompt_file}"
    )
    
    result = subprocess.run(["adb", "shell", adb_cmd], capture_output=True, text=True)
    
    if result.returncode != 0:
        return None, result.stderr
        
    return result.stdout, result.stderr

def parse_summary_output(raw_output):
    """Extracts summary from the raw model output"""
    content = raw_output.replace('\n', ' ').replace('> ', '')
    
    # Try to find the tags
    match = re.search(r'\[summary begin\](.*?)\[/?summary end\]', content, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: Just try to get the text after the prompt (if echoed)
    if "tags{task end}:" in content:
        parts = content.split("tags{task end}:")
        if len(parts) > 1:
            return parts[-1].strip()[:1000] # Safe limit
            
    return content[-500:].strip() # Return last chunk if all else fails

def main():
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR)

    articles_file, highlights_file, output_json, model_file = get_user_config()
    
    print("\n  [•] Reading input files...")
    articles = read_file_sections(articles_file, r'={3,}\s*ARTICLE\s+(\d+)\s*={3,}')
    highlights = read_file_sections(highlights_file, r'={3,}\s*SUMMARY\s+(\d+)\s*={3,}')
    highlights_dict = {num: content for num, content in highlights}
    
    print(f"  [✓] Loaded {len(articles)} articles.")
    print("\n  [•] Starting Summarization...\n")
    
    results = []
    
    for idx, (article_num, article_content) in enumerate(articles):
        print(f"  Processing Article {article_num} ({idx+1}/{len(articles)})...", end="", flush=True)
        
        start_time = time.time()
        
        try:
            # 1. Push Prompt File
            remote_prompt = push_prompt_to_phone(article_content, article_num)
            
            # 2. Run Inference
            raw_out, logs = run_inference_on_phone(remote_prompt, model_file)
            
            duration = time.time() - start_time
            
            if raw_out:
                summary = parse_summary_output(raw_out)
                print(f" Done ({duration:.2f}s)")
            else:
                summary = "Error: Inference Failed"
                print(" Failed.")
                
            # Cleanup phone file
            subprocess.run(["adb", "shell", f"rm {remote_prompt}"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            print(f" Error: {e}")
            summary = f"Error: {e}"
            logs = str(e)
            raw_out = ""

        # Save Result
        results.append({
            "article_number": article_num,
            "highlight": highlights_dict.get(article_num, ""),
            "generated_summary": summary,
            "article_text": article_content,
            "generation_logs": logs,
            "raw_model_output": raw_out
        })
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Cleanup Local
    try:
        import shutil
        shutil.rmtree(LOCAL_TEMP_DIR)
    except: pass

    print("\n" + "="*60)
    print("  TASK COMPLETE")
    print(f"  • Output: {output_json}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()