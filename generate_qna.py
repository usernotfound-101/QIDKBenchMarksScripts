#!/usr/bin/env python3
"""
squad_batch_processor_safe.py

Complete batch processing system:
1. Generate multiple batches of 10 questions each
2. Run each batch through the model
3. Parse all results safely (ignoring instruction blocks)
"""

import os
import sys
import json
import hashlib
import re
import subprocess
import random
from datasets import load_dataset

# ----------- CONFIG -----------
MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"
DEVICE = "HTP0"
CRUN_CMD = "./crun-cli.sh"
BATCH_SIZE = 10
BATCHES_DIR = "batches"
OUTPUTS_DIR = "outputs"
FINAL_OUTPUT = "final_output.jsonl"
# ------------------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def ensure_dirs():
    os.makedirs(BATCHES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ------------------- GENERATE -------------------
# ------------------- GENERATE -------------------
def generate_batches(num_batches):
    ensure_dirs()
    
    print("\nLoading SQuAD validation split...")
    ds = load_dataset("squad", split="validation")

    # --------- Group by context (topics) ---------
    from collections import defaultdict
    context_to_questions = defaultdict(list)
    for item in ds:
        context = item["context"]
        context_to_questions[context].append(item)

    topics = list(context_to_questions.keys())

    print(f"Generating {num_batches} batches of {BATCH_SIZE} questions each...")
    all_metadata = []

    for batch_num in range(num_batches):
        # Pick a random topic for this batch
        topic = random.choice(topics)
        questions_in_topic = context_to_questions[topic]

        # Sample questions from this topic (without replacement)
        batch_items = random.sample(
            questions_in_topic,
            min(BATCH_SIZE, len(questions_in_topic))
        )

        batch_questions = []
        batch_expected = []

        batch_file = os.path.join(BATCHES_DIR, f"batch_{batch_num+1:03d}.txt")

        with open(batch_file, 'w', encoding='utf-8') as f:
            for item in batch_items:
                q = item.get("question", "").strip()
                context = item.get("context", "").strip()  # <-- include context
                expected = item.get("answers", {}).get("text", [""])[0]

                batch_questions.append(q)
                batch_expected.append(expected)

                formatted = (
                    f"{{question start}}{q}{{question end}} "
                    f"{{context start}}{context}{{context end}} "
                    f"{{requirement start}}Answer concisely and in a single sentence. "
                    f"Wrap your answer between [answer start] and [answer end] exactly THIS IS A REQUIREMENT YOU MUST FOLLOW, WRAP YOU ANSWER WITH THESE TAGS COMPULSARILY.{{requirement end}}\n"
                )
                f.write(formatted)

        metadata = {
            "batch_num": batch_num + 1,
            "topic": topic,
            "questions": batch_questions,
            "expected_answers": batch_expected
        }
        all_metadata.append(metadata)

        print(f"  ✓ Batch {batch_num+1:03d}: {len(batch_questions)} questions from topic -> {batch_file}")

    # Save metadata
    metadata_file = os.path.join(BATCHES_DIR, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Generated {num_batches} batches")
    print(f"✓ Metadata saved to: {metadata_file}")
    print(f"\nNext step: Run batches")
    print(f"  python3 {sys.argv[0]} run")



# ------------------- RUN -------------------
def run_batches():
    ensure_dirs()
    
    if not os.path.exists(BATCHES_DIR):
        print(f"Error: {BATCHES_DIR} directory not found!")
        return False
    
    batch_files = sorted(f for f in os.listdir(BATCHES_DIR) if f.startswith('batch_') and f.endswith('.txt'))
    
    if not batch_files:
        print("No batch files found!")
        return False
    
    print(f"\nFound {len(batch_files)} batches to process")
    
    env = os.environ.copy()
    env["M"] = MODEL
    env["D"] = DEVICE
    
    successful = 0
    failed = 0
    
    for batch_file in batch_files:
        batch_path = os.path.join(BATCHES_DIR, batch_file)
        batch_name = batch_file.replace('.txt', '')
        output_path = os.path.join(OUTPUTS_DIR, f"{batch_name}_output.txt")
        
        print(f"Processing {batch_file}...")
        
        try:
            with open(batch_path, 'r') as input_f, open(output_path, 'w') as output_f:
                subprocess.run([CRUN_CMD], stdin=input_f, stdout=output_f, stderr=subprocess.PIPE,
                               env=env, timeout=600)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"  ✓ Output saved to {output_path}")
                successful += 1
            else:
                print(f"  ✗ No output generated")
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout (batch took > 10 minutes)")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Batch processing complete: Successful={successful}, Failed={failed}")
    return successful > 0

# ------------------- SAFE PARSER -------------------
def remove_curly_blocks(text: str) -> str:
    """
    Remove all {...} blocks sequentially, ignoring nested braces.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            j = text.find('}', i + 1)
            if j == -1:
                break
            i = j + 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)

def parse_batch_content(content, metadata):
    """
    Parse a single batch output file and safely extract answer blocks.
    Cleans content similarly to the standalone script:
      - Remove all newlines
      - Remove '> ' prompts
      - Remove {question…}, {context…}, and {requirement…} blocks entirely
      - Remove 'EOF by user'
      - Add newlines around { } and [ ] (optional for readability)
      - Collapse multiple spaces
      - Extract [answer start] ... [answer end] blocks in order
    """
    questions = metadata['questions']
    expected_answers = metadata['expected_answers']

    # 1. Remove all newlines
    content = content.replace('\n', '')

    # 2. Remove '> ' prompts
    content = content.replace('> ', '')

    # 3. Remove 'EOF by user'
    content = re.sub(r'EOF by user.*', '', content, flags=re.IGNORECASE)

    # 4. Remove {question}, {context}, {requirement} blocks
    content = re.sub(r'\{question start\}.*?\{question end\}', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\{context start\}.*?\{context end\}', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\{requirement start\}.*?\{requirement end\}', '', content, flags=re.IGNORECASE)

    # 5. Optional: add newline before/after { } and [ ] for readability
    content = content.replace('{', '\n{').replace('}', '}\n')
    content = content.replace('[', '\n[').replace(']', ']\n')

    # 6. Collapse multiple spaces
    content = re.sub(r'\s+', ' ', content)

    # 7. Extract all [answer start] ... [answer end] blocks
    answer_blocks = re.findall(r'\[answer start\].*?\[/?answer end\]', content, flags=re.IGNORECASE)

    results = []
    for i, q in enumerate(questions):
        if i < len(answer_blocks):
            answer = answer_blocks[i].strip()
        else:
            answer = "[NOT FOUND]"
        results.append({
            "question": q,
            "answer": answer,
            "expected": expected_answers[i],
            "hash": sha256(q + expected_answers[i])
        })

    return results

def parse_batches():
    ensure_dirs()
    metadata_file = os.path.join(BATCHES_DIR, "metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found!")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    output_files = sorted(f for f in os.listdir(OUTPUTS_DIR) if f.endswith('_output.txt'))
    
    if not output_files:
        print("No output files found!")
        return
    
    all_results = []
    
    for output_file in output_files:
        match = re.search(r'batch_(\d+)_output\.txt', output_file)
        if not match:
            continue
        batch_num = int(match.group(1))
        metadata = next((m for m in all_metadata if m['batch_num']==batch_num), None)
        if not metadata:
            continue
        
        output_path = os.path.join(OUTPUTS_DIR, output_file)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = parse_batch_content(content, metadata)
        all_results.extend(results)
    
    # Write final JSONL
    with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"\nParsing complete. Output: {FINAL_OUTPUT}")
    print(f"Total Q&A pairs: {len(all_results)}")

# ------------------- MAIN -------------------
def print_usage():
    print("SQuAD Batch Processor Safe")
    print("=" * 70)
    print(f"Usage:")
    print(f"  {sys.argv[0]} generate <num_batches>  - Generate batch files")
    print(f"  {sys.argv[0]} run                     - Run all batches through model")
    print(f"  {sys.argv[0]} parse                   - Parse results into JSONL")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "generate":
        if len(sys.argv) < 3:
            print("Specify number of batches")
            sys.exit(1)
        generate_batches(int(sys.argv[2]))
    elif cmd == "run":
        if not run_batches():
            sys.exit(1)
    elif cmd == "parse":
        parse_batches()
    else:
        print(f"Unknown command '{cmd}'")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
