#!/usr/bin/env python3
"""
Evaluate multiple SLMs on SQuAD dataset with BERT score using crun-cli.sh
This version batches questions to avoid reloading the model.
"""

import subprocess
import sys
import os
import json
import random
import hashlib
import re
from pathlib import Path
from datasets import load_dataset
from bert_score import score as bert_score
import pandas as pd

# Models to evaluate
MODELS = [
    "qwen3-4b-instruct-2507-q4km.gguf",
    "Llama-3.2-1B-Instruct-Q4_0.gguf",
    "gemma-3-4b-it-q4_0.gguf",
    "phi-3.5-mini-instruct.Q4_K_M.gguf"
]

DEVICE = "HTP0"
NUM_QUESTIONS = 10
RANDOM_SEED = 42
BATCH_SIZE = 5  # Process 5 questions per model load

# JSON schema for structured output with hash tracking
JSON_SCHEMA = '{"type":"object","properties":{"hash":{"type":"string"},"answer":{"type":"string","minLength":1,"maxLength":50}},"required":["hash","answer"]}'

def get_question_hash(question):
    """Generate unique hash for a question"""
    return hashlib.md5(question.encode()).hexdigest()[:8]

def load_squad_questions(num_questions=10, seed=42):
    """Load random questions from SQuAD dataset"""
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation")
    
    # Set seed for reproducibility
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_questions)
    
    questions = []
    for idx in indices:
        item = dataset[idx]
        questions.append({
            'id': item['id'],
            'question': item['question'],
            'context': item['context'],
            'answers': item['answers']['text'],
            'hash': get_question_hash(item['question'])
        })
    
    return questions

def create_output_dirs(model_name):
    """Create response and log directories for a model"""
    model_slug = model_name.replace('.gguf', '').replace('.', '_')
    response_dir = Path(f"responses_{model_slug}")
    log_dir = Path(f"logs_{model_slug}")
    
    response_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    return response_dir, log_dir

def extract_answers_from_output(output_text, expected_hashes):
    """Extract answers from model output with hash tracking"""
    answers = {}
    
    # Method 1: Find all JSON objects with flexible regex
    json_pattern = r'\{\s*"hash"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]*)"\s*\}'
    matches = re.finditer(json_pattern, output_text, re.DOTALL)
    
    for match in matches:
        hash_val = match.group(1)
        answer = match.group(2)
        if hash_val in expected_hashes:
            answers[hash_val] = answer
    
    # Method 2: Try reversed order (answer before hash)
    if not answers:
        json_pattern2 = r'\{\s*"answer"\s*:\s*"([^"]*)"\s*,\s*"hash"\s*:\s*"([^"]+)"\s*\}'
        matches = re.finditer(json_pattern2, output_text, re.DOTALL)
        for match in matches:
            answer = match.group(1)
            hash_val = match.group(2)
            if hash_val in expected_hashes:
                answers[hash_val] = answer
    
    # Method 3: Line-by-line JSON parsing
    if not answers:
        for line in output_text.split('\n'):
            line = line.strip()
            if '{' in line and '"hash"' in line and '"answer"' in line:
                try:
                    # Extract just the JSON part
                    start = line.find('{')
                    end = line.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = line[start:end]
                        data = json.loads(json_str)
                        if 'hash' in data and 'answer' in data:
                            if data['hash'] in expected_hashes:
                                answers[data['hash']] = data['answer']
                except (json.JSONDecodeError, ValueError):
                    continue
    
    return answers

def run_batch(model, question_batch, response_dir, log_dir, batch_num):
    """Run crun-cli.sh with a batch of questions"""
    
    print(f"\n  Batch {batch_num}: Processing {len(question_batch)} questions...")
    
    # Build the command with all questions using proper escaping
    expected_hashes = set()
    
    # Create a temporary file with all prompts to avoid shell escaping issues
    prompts_file = response_dir / f"batch_{batch_num}_prompts.txt"
    with open(prompts_file, 'w') as f:
        for q_data in question_batch:
            q_hash = q_data['hash']
            expected_hashes.add(q_hash)
            # Simple prompt without extra quotes
            prompt = f'Answer this question with hash "{q_hash}": {q_data["question"]}'
            f.write(prompt + '\n')
    
    # Build command that reads prompts from file
    # Use a shell script to properly handle each prompt
    cmd_script = f'''#!/bin/bash
M={model} D={DEVICE} ./crun-cli.sh -no-cnv \\
  --json-schema '{JSON_SCHEMA}' \\
  -n 75 \\
'''
    
    # Add each prompt as a separate -p argument with proper escaping
    with open(prompts_file, 'r') as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                # Escape for shell: replace ' with '\'' and wrap in single quotes
                escaped = prompt.replace("'", "'\\''")
                cmd_script += f"  -p '{escaped}' \\\n"
    
    # Remove trailing backslash and newline
    cmd_script = cmd_script.rstrip(' \\\n')
    
    # Save script to file
    script_file = response_dir / f"batch_{batch_num}_run.sh"
    with open(script_file, 'w') as f:
        f.write(cmd_script)
    
    # Make executable
    os.chmod(script_file, 0o755)
    
    output_file = response_dir / f"batch_{batch_num}_output.txt"
    log_file = log_dir / f"batch_{batch_num}_log.txt"
    
    print(f"    Running command (see {log_file})...")
    
    try:
        # Run the script
        result = subprocess.run(
            f'bash {script_file}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per batch
        )
        
        # Save raw output
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        
        # Save log
        with open(log_file, 'w') as f:
            f.write(f"=== Script File ===\n{script_file}\n\n")
            f.write(f"=== Prompts ===\n")
            with open(prompts_file, 'r') as pf:
                f.write(pf.read())
            f.write(f"\n=== Return Code ===\n{result.returncode}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n")
        
        if result.returncode != 0:
            print(f"    ⚠ Warning: Non-zero return code ({result.returncode})")
            # Check stderr for the actual error
            if "error:" in result.stderr.lower():
                error_lines = [line for line in result.stderr.split('\n') if 'error:' in line.lower()]
                print(f"    Error: {error_lines[0] if error_lines else 'Unknown'}")
        
        # Extract answers
        answers = extract_answers_from_output(result.stdout, expected_hashes)
        
        # Save individual responses
        for q_data in question_batch:
            q_hash = q_data['hash']
            answer = answers.get(q_hash, "")
            
            response_file = response_dir / f"{q_hash}_response.txt"
            with open(response_file, 'w') as f:
                f.write(f"Question: {q_data['question']}\n")
                f.write(f"Hash: {q_hash}\n")
                f.write(f"Answer: {answer}\n\n")
                f.write(f"Reference Answers: {q_data['answers']}\n")
            
            question_file = response_dir / f"{q_hash}_question.json"
            with open(question_file, 'w') as f:
                json.dump(q_data, f, indent=2)
        
        print(f"    ✓ Extracted {len(answers)}/{len(question_batch)} answers")
        
        return answers
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after 120 seconds")
        with open(log_file, 'w') as f:
            f.write(f"ERROR: Timeout after 120 seconds\n")
            f.write(f"Script: {script_file}\n")
        return {}
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        with open(log_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"Script: {script_file}\n")
        return {}

def run_model_batched(model, questions, response_dir, log_dir):
    """Run model with questions in batches to minimize reloading"""
    
    print(f"Processing {len(questions)} questions in batches of {BATCH_SIZE}...")
    
    all_responses = {}
    
    # Split into batches
    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        answers = run_batch(model, batch, response_dir, log_dir, batch_num)
        all_responses.update(answers)
    
    return all_responses

def evaluate_bert_scores(questions, model_responses):
    """Calculate BERT scores for all models"""
    print("\n" + "="*60)
    print("=== Calculating BERT Scores ===")
    print("="*60)
    
    results = []
    
    for model_name, responses in model_responses.items():
        print(f"\nEvaluating {model_name}...")
        
        candidates = []
        references = []
        
        for q in questions:
            q_hash = q['hash']
            candidate = responses.get(q_hash, "")
            # Use first answer as reference
            reference = q['answers'][0] if q['answers'] else ""
            
            candidates.append(candidate)
            references.append(reference)
        
        answered = len([c for c in candidates if c])
        print(f"  Answers collected: {answered}/{len(candidates)}")
        
        # Calculate BERT score only if we have answers
        if candidates and references and answered > 0:
            try:
                P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
                
                avg_precision = P.mean().item()
                avg_recall = R.mean().item()
                avg_f1 = F1.mean().item()
                
                results.append({
                    'model': model_name,
                    'answered': answered,
                    'total': len(candidates),
                    'coverage': f"{answered}/{len(candidates)}",
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1
                })
                
                print(f"  Precision: {avg_precision:.4f}")
                print(f"  Recall:    {avg_recall:.4f}")
                print(f"  F1 Score:  {avg_f1:.4f}")
            except Exception as e:
                print(f"  ⚠ BERT score calculation failed: {e}")
                print(f"  Falling back to simple metrics...")
                
                # Fallback: simple exact match and length-based scoring
                exact_matches = sum(1 for c, r in zip(candidates, references) if c.strip().lower() == r.strip().lower())
                partial_matches = sum(1 for c, r in zip(candidates, references) if c and r and any(word in c.lower() for word in r.lower().split()))
                
                # Simple F1-like metric based on matches
                precision = partial_matches / answered if answered > 0 else 0
                recall = partial_matches / len(references) if references else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'model': model_name,
                    'answered': answered,
                    'total': len(candidates),
                    'coverage': f"{answered}/{len(candidates)}",
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'note': 'fallback_metrics'
                })
                
                print(f"  Exact matches: {exact_matches}")
                print(f"  Partial matches: {partial_matches}")
                print(f"  Simple F1: {f1:.4f}")
        else:
            print(f"  ✗ No valid responses to evaluate")
            results.append({
                'model': model_name,
                'answered': 0,
                'total': len(candidates),
                'coverage': f"0/{len(candidates)}",
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            })
    
    return results

def main():
    print("="*60)
    print("=== SQuAD Model Evaluator (crun-cli Batched) ===")
    print("="*60)
    print()
    
    # Load questions
    questions = load_squad_questions(NUM_QUESTIONS, RANDOM_SEED)
    print(f"Loaded {len(questions)} questions from SQuAD")
    print(f"Batch size: {BATCH_SIZE} questions per model load\n")
    
    # Save questions to file for reference
    with open('evaluation_questions.json', 'w') as f:
        json.dump(questions, f, indent=2)
    print("✓ Questions saved to evaluation_questions.json\n")
    
    # Store all model responses
    model_responses = {}
    
    # Evaluate each model
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {model}")
        print('='*60)
        
        # Create directories
        response_dir, log_dir = create_output_dirs(model)
        print(f"Response dir: {response_dir}")
        print(f"Log dir:      {log_dir}")
        
        # Run model with batched questions
        responses = run_model_batched(model, questions, response_dir, log_dir)
        model_responses[model] = responses
        
        print(f"\n✓ Model complete: {len(responses)}/{len(questions)} answers collected")
    
    if not any(model_responses.values()):
        print("\n✗ No responses collected from any model. Check logs for errors.")
        return
    
    # Evaluate BERT scores
    print("\n" + "="*60)
    bert_results = evaluate_bert_scores(questions, model_responses)
    
    if not bert_results:
        print("\n✗ No results to save")
        return
    
    # Save results
    results_df = pd.DataFrame(bert_results)
    results_df = results_df.sort_values('f1', ascending=False)
    results_df.to_csv('bert_scores.csv', index=False)
    
    print("\n" + "="*60)
    print("=== Final Results (sorted by F1) ===")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\n✓ Results saved to bert_scores.csv")
    
    # Save detailed results
    detailed_results = {
        'questions': questions,
        'model_responses': model_responses,
        'bert_scores': bert_results
    }
    
    with open('detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("✓ Detailed results saved to detailed_results.json")

if __name__ == "__main__":
    main()
