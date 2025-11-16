#!/usr/bin/env python3
"""
Process articles and highlights, generate summaries using SLM, and output JSON.
"""

import re
import json
import subprocess
import sys
import os


def read_file_sections(filepath, delimiter_pattern):
    r"""
    Read a file and split it into sections based on delimiter pattern.
    
    Args:
        filepath: Path to the input file
        delimiter_pattern: Regex pattern for section delimiters (e.g., r'={3,}\s*ARTICLE\s+(\d+)\s*={3,}')
    
    Returns:
        List of tuples: [(section_number, section_content), ...]
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    
    # Split by section headers
    sections = re.split(delimiter_pattern + r'\s*\n', content)
    
    # Remove empty first element if file starts with delimiter
    if sections[0].strip() == '':
        sections = sections[1:]
    
    # Process sections in pairs (number, content)
    result = []
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            section_num = sections[i]
            section_content = sections[i + 1].strip()
            if section_content:
                result.append((section_num, section_content))
    
    return result


def generate_summary_with_slm(article_text):
    """
    Generate summary using the SLM via crun-cli.sh
    Uses M and D environment variables
    
    Args:
        article_text: The article text to summarize
    
    Returns:
        Tuple of (summary_text, logs)
    """
    # Escape single quotes in the article text for shell
    escaped_text = article_text.replace("'", "'\\''")
    
    # Create the prompt requesting tagged output for easy parsing
    prompt = f'"Summarize this article briefly and wrap your summary with [summary begin] and [summary end] tags: {escaped_text}"'
    
    # Build the command with token limit to force shorter responses
    # -n 300 limits output to ~200 words (1.5 tokens per word average)
    cmd = f"./crun-cli.sh -no-cnv -n 300 -p '{prompt}'"
    
    try:
        # Run the command and capture output - NO TIMEOUT
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ.copy()  # Pass environment variables including M and D
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            logs = result.stderr.strip()  # Capture stderr logs
            
            # Parse the summary between [summary begin] and [summary end] tags
            begin_tag = "[summary begin]"
            end_tag = "[summary end]"
            
            if begin_tag in output and end_tag in output:
                # Extract only the content between the tags
                start_idx = output.find(begin_tag) + len(begin_tag)
                end_idx = output.find(end_tag)
                summary = output[start_idx:end_idx].strip()
                return summary, logs
            else:
                # Fallback: if tags not found, return the whole output
                print(f"  Warning: Summary tags not found in output, using full response")
                return output.strip(), logs
        else:
            error_msg = f"Error running SLM: {result.stderr}"
            print(error_msg)
            return "Error generating summary", result.stderr
    
    except Exception as e:
        error_msg = f"Error running SLM: {e}"
        print(error_msg)
        return "Error generating summary", str(e)


def main():
    # Configuration
    if len(sys.argv) < 3:
        print("Usage: python article_summarizer.py <articles_file> <highlights_file> [output_json]")
        print("Example: M=gemma-3-4b-it-q4_0.gguf D=HTP0 python article_summarizer.py articles.txt highlights.txt")
        print("\nNote: Set M and D environment variables before running")
        sys.exit(1)
    
    articles_file = sys.argv[1]
    highlights_file = sys.argv[2]
    output_json = sys.argv[3] if len(sys.argv) > 3 else 'summaries_output.json'
    
    # Check if M environment variable is set
    if 'M' not in os.environ:
        print("Error: M environment variable not set. Please set it to your model path.")
        print("Example: export M=gemma-3-4b-it-q4_0.gguf")
        sys.exit(1)
    
    if 'D' not in os.environ:
        print("Error: D environment variable not set. Please set it to your device.")
        print("Example: export D=HTP0")
        sys.exit(1)
    
    print(f"Using model: {os.environ['M']}")
    print(f"Using device: {os.environ['D']}")
    print()
    
    # Read articles and highlights
    print("Reading articles...")
    articles = read_file_sections(articles_file, r'={3,}\s*ARTICLE\s+(\d+)\s*={3,}')
    
    print("Reading highlights...")
    highlights = read_file_sections(highlights_file, r'={3,}\s*SUMMARY\s+(\d+)\s*={3,}')
    
    # Create a dictionary for highlights for easy lookup
    highlights_dict = {num: content for num, content in highlights}
    
    # Process each article
    results = []
    for article_num, article_content in articles:
        print(f"\nProcessing Article {article_num}...")
        
        # Generate summary using SLM
        print(f"  Generating summary with SLM...")
        summary, logs = generate_summary_with_slm(article_content)
        
        # Get corresponding highlight
        highlight = highlights_dict.get(article_num, "No highlight available")
        
        # Create result object with logs
        result = {
            "article_number": article_num,
            "highlight": highlight,
            "generated_summary": summary,
            "article_text": article_content,
            "generation_logs": logs
        }
        
        results.append(result)
        print(f"  ✓ Article {article_num} processed")
    
    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ All done! Results saved to: {output_json}")
    print(f"  Total articles processed: {len(results)}")


if __name__ == '__main__':
    main()