#!/usr/bin/env python3
"""
Evaluate multiple SLMs on SQuAD dataset with BERT score
"""

import subprocess
import sys
import os
import json
import random
import hashlib
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

def run_model(model, question_data, response_file, log_file):
    """Run brun-cli.sh for a single question"""
    
    # Create prompt with 20-word constraint and explicit instruction
    prompt = f"Answer in EXACTLY 20 words or less. Be concise: {question_data['question']}"
    
    # Escape the prompt properly for shell
    escaped_prompt = prompt.replace('"', '\\"')
    
    # Construct the command with log file redirection
    # Add -n 50 to limit max tokens generated
    cmd = (
        f'M={model} D={DEVICE} ./brun-cli.sh '
        f'-no-cnv -p "\\"{escaped_prompt}\\"" '
        f'-r \'"<|im_end|>"\' -n 50 >| {response_file}'
    )
    
    print(f"  Running: {question_data['hash']}")
    
    try:
        # Execute the command with timeout (60 seconds per question)
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            check=False,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Save log output
        with open(log_file, 'w') as f:
            f.write(f"=== Command ===\n{cmd}\n\n")
            f.write(f"=== Return Code ===\n{result.returncode}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"  ⏱ Timeout - killing process")
        with open(log_file, 'w') as f:
            f.write(f"=== Command ===\n{cmd}\n\n")
            f.write(f"ERROR: Timeout after 60 seconds\n")
        return False
        
    except Exception as e:
        print(f"Error executing command: {e}")
        with open(log_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
        return False

def extract_response(response_file):
    """Extract the actual response from the output file"""
    try:
        with open(response_file, 'r') as f:
            content = f.read()
        
        # Clean up the response - remove extra newlines and repetition
        lines = content.strip().split('\n')
        
        # Take only the first substantial line as the response
        # Skip empty lines and the original prompt echo
        response_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Answer in'):
                response_lines.append(line)
                # Stop after first good response (before repetition starts)
                if len(response_lines) >= 1:
                    break
        
        response = ' '.join(response_lines)
        
        # Truncate at common end markers
        for marker in ['[end of text]', '(Word count:', 'Note:', 'Corrected', 'The prize is']:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Limit to roughly 20 words
        words = response.split()
        if len(words) > 25:  # Allow slight overflow
            response = ' '.join(words[:25])
        
        return response
        
    except Exception as e:
        print(f"Error reading response: {e}")
        return ""

def evaluate_bert_scores(questions, model_responses):
    """Calculate BERT scores for all models"""
    print("\n=== Calculating BERT Scores ===")
    
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
        
        # Calculate BERT score
        if candidates and references:
            P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
            
            avg_precision = P.mean().item()
            avg_recall = R.mean().item()
            avg_f1 = F1.mean().item()
            
            results.append({
                'model': model_name,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            })
            
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1: {avg_f1:.4f}")
    
    return results

def main():
    print("=== SQuAD Model Evaluator ===\n")
    
    # Load questions
    questions = load_squad_questions(NUM_QUESTIONS, RANDOM_SEED)
    print(f"Loaded {len(questions)} questions from SQuAD\n")
    
    # Save questions to file for reference
    with open('evaluation_questions.json', 'w') as f:
        json.dump(questions, f, indent=2)
    print("Questions saved to evaluation_questions.json\n")
    
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
        print(f"Log dir: {log_dir}\n")
        
        model_responses[model] = {}
        
        # Run each question
        for i, q_data in enumerate(questions, 1):
            print(f"Question {i}/{len(questions)}: {q_data['hash']}")
            
            response_file = response_dir / f"{q_data['hash']}_response.txt"
            log_file = log_dir / f"{q_data['hash']}_log.txt"
            
            # Save question context
            question_file = response_dir / f"{q_data['hash']}_question.json"
            with open(question_file, 'w') as f:
                json.dump(q_data, f, indent=2)
            
            # Run model
            success = run_model(model, q_data, response_file, log_file)
            
            if success:
                response = extract_response(response_file)
                model_responses[model][q_data['hash']] = response
                print(f"  ✓ Success")
            else:
                print(f"  ✗ Failed")
                model_responses[model][q_data['hash']] = ""
    
    # Evaluate BERT scores
    bert_results = evaluate_bert_scores(questions, model_responses)
    
    # Save results
    results_df = pd.DataFrame(bert_results)
    results_df.to_csv('bert_scores.csv', index=False)
    print("\n=== Results saved to bert_scores.csv ===")
    print(results_df.to_string(index=False))
    
    # Save detailed results
    detailed_results = {
        'questions': questions,
        'model_responses': model_responses,
        'bert_scores': bert_results
    }
    
    with open('detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print("\nDetailed results saved to detailed_results.json")

if __name__ == "__main__":
    main()
