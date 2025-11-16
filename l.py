#!/usr/bin/env python3
"""
Split a text file containing multiple articles into separate files.
Articles are delimited by headers like "=== ARTICLE N ==="
"""

import re
import os
import sys


def split_articles(input_file, output_dir='articles'):
    """
    Split articles from input file into separate files.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save the split articles (default: 'articles')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Split by article headers using regex
    # Pattern matches "=== ARTICLE N ===" where N is a number
    articles = re.split(r'={3,}\s*ARTICLE\s+(\d+)\s*={3,}\s*\n', content)
    
    # Remove empty first element if file starts with a delimiter
    if articles[0].strip() == '':
        articles = articles[1:]
    
    # Process articles in pairs (number, content)
    article_count = 0
    for i in range(0, len(articles), 2):
        if i + 1 < len(articles):
            article_num = articles[i]
            article_content = articles[i + 1].strip()
            
            if article_content:  # Only save non-empty articles
                output_file = os.path.join(output_dir, f'article_{article_num}.txt')
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(article_content)
                
                article_count += 1
                print(f"Saved: {output_file}")
    
    print(f"\nTotal articles extracted: {article_count}")
    print(f"Articles saved to: {output_dir}/")


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python article_splitter.py <input_file> [output_directory]")
        print("Example: python article_splitter.py articles.txt")
        print("Example: python article_splitter.py articles.txt output_folder")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'articles'
    
    split_articles(input_file, output_dir)