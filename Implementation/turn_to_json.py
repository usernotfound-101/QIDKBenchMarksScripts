import json
import re
from pathlib import Path

INPUT_FILE = Path(__file__).with_name("questions.txt")
OUTPUT_FILE = Path(__file__).with_name("questions.json")

def parse_questions(text: str):
    # Matches lines like: [Question 1]
    header_re = re.compile(r'^\[Question\s+(\d+)\]\s*$', re.IGNORECASE)

    questions = []
    current_number = None
    current_lines = []

    for line in text.splitlines():
        header_match = header_re.match(line.strip())
        if header_match:
            # flush previous question
            if current_number is not None and current_lines:
                question_text = "\n".join(current_lines).strip()
                questions.append({"number": current_number, "question": question_text})
            # start new question
            current_number = int(header_match.group(1))
            current_lines = []
        else:
            # accumulate question text
            if current_number is not None:
                current_lines.append(line.rstrip())

    # flush last question
    if current_number is not None and current_lines:
        question_text = "\n".join(current_lines).strip()
        questions.append({"number": current_number, "question": question_text})

    return questions

def main():
    text = INPUT_FILE.read_text(encoding="utf-8")
    questions = parse_questions(text)

    # Optional: de-duplicate by (number, question) if needed
    # seen = set()
    # unique = []
    # for q in questions:
    #     key = (q["number"], q["question"])
    #     if key not in seen:
    #         seen.add(key)
    #         unique.append(q)
    # questions = unique

    OUTPUT_FILE.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()