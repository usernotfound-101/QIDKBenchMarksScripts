#!/usr/bin/env python3
"""
multi_model_squad_processor.py

Automated batch processing system for multiple models:
1. Generate batches of questions
2. Run each model on all batches
3. Parse results for each model into separate directories

Usage:
  python3 multi_model_squad_processor.py <num_batches>
"""

import os
import sys
import json
import hashlib
import re
import subprocess
import random
import time
from datasets import load_dataset

# ----------- CONFIG -----------
MODELS = [
    "Llama-3.2-1B-Instruct-Q4_0.gguf",
    "gemma-3-4b-it-q4_0.gguf",
    "phi-3.5-mini-instruct.Q4_K_M.gguf",
    "qwen3-4b-instruct-2507-q4km.gguf"
]
DEVICE = "HTP0"
CRUN_CMD = "./crun-cli.sh"
BATCH_SIZE = 10
BASE_DIR = "squad_results"
BATCHES_DIR = "batches"
# ------------------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def get_model_dirname(model_name):
    """Convert model filename to directory name."""
    # Remove .gguf extension and replace special chars
    return model_name.replace('.gguf', '').replace('.', '_').replace('-', '_')

def ensure_dirs(model_name=None):
    """Create necessary directories."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, BATCHES_DIR), exist_ok=True)
    
    if model_name:
        model_dir = os.path.join(BASE_DIR, get_model_dirname(model_name))
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, "outputs"), exist_ok=True)

def generate_batches(num_batches):
    """Generate multiple batch files with questions."""
    ensure_dirs()
    
    print(f"\nLoading SQuAD validation split...")
    ds = load_dataset("squad", split="validation")
    
    total_questions = num_batches * BATCH_SIZE
    total_available = len(ds)
    to_process = min(total_questions, total_available)
    
    if to_process < total_questions:
        print(f"Note: Only {total_available} examples available.")
        num_batches = to_process // BATCH_SIZE
    
    print(f"Generating {num_batches} batches of {BATCH_SIZE} questions each...")
    
    # Create random indices for sampling
    all_indices = list(range(total_available))
    random.shuffle(all_indices)
    selected_indices = all_indices[:to_process]
    
    # Store metadata for all batches
    all_metadata = []
    
    batches_path = os.path.join(BASE_DIR, BATCHES_DIR)
    
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, to_process)
        
        batch_questions = []
        batch_expected = []
        
        batch_file = os.path.join(batches_path, f"batch_{batch_num+1:03d}.txt")
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for i in range(start_idx, end_idx):
                idx = selected_indices[i]
                item = ds[idx]
                q = item.get("question", "").strip()
                
                # Extract expected answer
                expected = ""
                if "answers" in item and isinstance(item["answers"], dict):
                    txts = item["answers"].get("text", [])
                    expected = txts[0] if txts else ""
                
                batch_questions.append(q)
                batch_expected.append(expected)
                
                # Format with clear delimiters
                formatted = f'{{question start}}{q}{{question end}} {{requirement start}}Answer in one sentence with [answer start] at beginning and [answer end] at end.{{requirement end}}\n'
                f.write(formatted)
        
        # Save metadata
        metadata = {
            "batch_num": batch_num + 1,
            "questions": batch_questions,
            "expected_answers": batch_expected
        }
        all_metadata.append(metadata)
        
        print(f"  ✓ Batch {batch_num+1:03d}: {len(batch_questions)} questions -> {batch_file}")
    
    # Save all metadata
    metadata_file = os.path.join(batches_path, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Generated {num_batches} batches with random questions")
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return num_batches

def run_model_batches(model_name, num_batches):
    """Run all batches through a specific model."""
    ensure_dirs(model_name)
    
    batches_path = os.path.join(BASE_DIR, BATCHES_DIR)
    model_dir = os.path.join(BASE_DIR, get_model_dirname(model_name))
    outputs_dir = os.path.join(model_dir, "outputs")
    
    # Get all batch files
    batch_files = sorted([f for f in os.listdir(batches_path) if f.startswith('batch_') and f.endswith('.txt')])
    
    if not batch_files:
        print(f"  ✗ No batch files found!")
        return False
    
    print(f"\n{'='*70}")
    print(f"Processing model: {model_name}")
    print(f"Device: {DEVICE}, Batches: {len(batch_files)}")
    print(f"{'='*70}")
    
    env = os.environ.copy()
    env["M"] = model_name
    env["D"] = DEVICE
    
    successful = 0
    failed = 0
    
    for batch_file in batch_files:
        batch_path = os.path.join(batches_path, batch_file)
        batch_name = batch_file.replace('.txt', '')
        output_path = os.path.join(outputs_dir, f"{batch_name}_output.txt")
        
        print(f"  Processing {batch_file}...", end=' ', flush=True)
        
        try:
            # Run: cat batch_file | ./crun-cli.sh > output
            with open(batch_path, 'r') as input_f:
                with open(output_path, 'w') as output_f:
                    proc = subprocess.run(
                        [CRUN_CMD],
                        stdin=input_f,
                        stdout=output_f,
                        stderr=subprocess.PIPE,
                        env=env,
                        timeout=180  # 3 minute timeout per batch
                    )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✓")
                successful += 1
            else:
                print(f"✗ (no output)")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"✗ (timeout)")
            failed += 1
        except Exception as e:
            print(f"✗ ({str(e)[:30]})")
            failed += 1
    
    print(f"\n  Results: {successful} successful, {failed} failed")
    return successful > 0

def parse_batch_content(content, metadata):
    """Parse a single batch output file."""
    questions = metadata['questions']
    expected_answers = metadata['expected_answers']
    
    # Step 1: Remove all "> " prompts
    content = re.sub(r'>\s+', '', content)
    
    # Step 2: Remove "EOF by user"
    content = re.sub(r'EOF by user.*', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 3: Remove requirement blocks
    content = re.sub(r'\{requirement start\}.*?\{requirement end\}', '', content, flags=re.IGNORECASE)
    
    # Step 4: Remove newlines
    content = content.replace('\n', '')
    
    # Step 5: Normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    results = []
    
    # Extract all answer blocks
    answer_pattern = r'\[answer start\](.*?)\[answer end\]'
    answer_blocks = re.findall(answer_pattern, content, re.DOTALL | re.IGNORECASE)
    
    # Match answers with questions
    for i, q in enumerate(questions):
        if i < len(answer_blocks):
            # Clean up answer
            answer = answer_blocks[i].strip()
            
            # Remove any echoed question tags
            answer = re.sub(r'\{question start\}.*?\{question end\}', '', answer, flags=re.IGNORECASE)
            
            # Normalize whitespace again
            cleaned_answer = ' '.join(answer.split())
        else:
            cleaned_answer = "[NOT FOUND]"
        
        # Create result record
        results.append({
            "question": q,
            "answer": cleaned_answer,
            "expected": expected_answers[i],
            "hash": sha256(q + expected_answers[i])
        })
    
    return results

def parse_model_outputs(model_name):
    """Parse all output files for a specific model and create JSONL."""
    model_dir = os.path.join(BASE_DIR, get_model_dirname(model_name))
    outputs_dir = os.path.join(model_dir, "outputs")
    batches_path = os.path.join(BASE_DIR, BATCHES_DIR)
    
    # Load metadata
    metadata_file = os.path.join(batches_path, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"  ✗ Metadata file not found: {metadata_file}")
        return False
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    # Get all output files
    if not os.path.exists(outputs_dir):
        print(f"  ✗ Outputs directory not found: {outputs_dir}")
        return False
    
    output_files = sorted([f for f in os.listdir(outputs_dir) if f.endswith('_output.txt')])
    
    if not output_files:
        print(f"  ✗ No output files found!")
        return False
    
    print(f"  Parsing {len(output_files)} output files...")
    
    all_results = []
    total_found = 0
    total_expected = 0
    
    for output_file in output_files:
        # Extract batch number from filename
        match = re.search(r'batch_(\d+)_output\.txt', output_file)
        if not match:
            continue
        
        batch_num = int(match.group(1))
        
        # Find corresponding metadata
        metadata = None
        for m in all_metadata:
            if m['batch_num'] == batch_num:
                metadata = m
                break
        
        if not metadata:
            continue
        
        output_path = os.path.join(outputs_dir, output_file)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract Q&A pairs
        results = parse_batch_content(content, metadata)
        
        all_results.extend(results)
        total_found += len([r for r in results if r['answer'] and not r['answer'].startswith('[')])
        total_expected += len(metadata['questions'])
    
    # Write final output
    output_jsonl = os.path.join(model_dir, "results.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Results saved to: {output_jsonl}")
    print(f"  ✓ Valid answers: {total_found}/{total_expected}")
    
    return True

def process_all_models(num_batches):
    """Main function to process all models."""
    start_time = time.time()
    
    print("="*70)
    print("MULTI-MODEL SQUAD BATCH PROCESSOR")
    print("="*70)
    
    # Step 1: Generate batches
    print("\n[STEP 1/3] Generating batches...")
    num_batches = generate_batches(num_batches)
    
    # Step 2: Run each model
    print("\n[STEP 2/3] Running models...")
    model_results = {}
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[Model {i}/{len(MODELS)}]")
        success = run_model_batches(model, num_batches)
        model_results[model] = success
    
    # Step 3: Parse results
    print("\n[STEP 3/3] Parsing results...")
    for i, model in enumerate(MODELS, 1):
        print(f"\n[Model {i}/{len(MODELS)}] {model}")
        if model_results[model]:
            parse_model_outputs(model)
        else:
            print(f"  ✗ Skipping (no successful batches)")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Models processed: {len(MODELS)}")
    print(f"\nResults directory: {BASE_DIR}/")
    print("\nPer-model results:")
    for model in MODELS:
        model_dir = get_model_dirname(model)
        jsonl_path = os.path.join(BASE_DIR, model_dir, "results.jsonl")
        if os.path.exists(jsonl_path):
            print(f"  ✓ {model_dir}/results.jsonl")
        else:
            print(f"  ✗ {model_dir}/ (failed)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 multi_model_squad_processor.py <num_batches>")
        print(f"\nConfigured models ({len(MODELS)}):")
        for model in MODELS:
            print(f"  - {model}")
        print(f"\nBatch size: {BATCH_SIZE} questions")
        print(f"Device: {DEVICE}")
        sys.exit(1)
    
    try:
        num_batches = int(sys.argv[1])
        if num_batches < 1:
            print("Error: Number of batches must be >= 1")
            sys.exit(1)
        
        process_all_models(num_batches)
        
    except ValueError:
        print("Error: Invalid number")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()