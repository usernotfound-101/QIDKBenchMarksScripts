import os
import json
import re
import sys
import time
import subprocess
from pathlib import Path

# --- discover models on device ---
result = subprocess.run(
    ["adb", "shell", "ls", "/data/local/tmp/gguf"],
    capture_output=True,
    text=True,
    check=True,
)
model_files = [m.strip() for m in result.stdout.splitlines() if m.strip()]

LLAMA_BIN_DIR = "/data/local/tmp/llama/build-clblast/bin"
LLAMA_CLI_PATH = os.path.join(LLAMA_BIN_DIR, "llama-cli")
MODEL_BASE_DIR = "/data/local/tmp/gguf"
REMOTE_PROMPT_FILE = "/data/local/tmp/prompt.txt"
REMOTE_RESULT_FILE = "/data/local/tmp/result.json"

# --- host-side paths ---
script_dir = Path(__file__).resolve().parent
questions_file_path = script_dir / "questions.json"
output_base_dir = script_dir / "outputs"
output_base_dir.mkdir(exist_ok=True)

# --- JSON schema expected from models ---
json_schema = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        }
    },
    "required": ["answers"],
}
json_schema_str = json.dumps(json_schema)


def extract_json_from_output(content: str):
    """Same strategy as generate_qna.py: try direct parse, then balanced braces, then regex."""
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
                candidate = content[start : i + 1]
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


def run_one_question(question: dict):
    """
    For a given question dict: {"number": int, "question": str}
    - create outputs/<number>/<model_name>.json
    - each file: {"inference_time_seconds": float, "response": <parsed or raw>}
    """
    q_num = question["number"]
    q_text = question["question"]

    # per-question directory
    question_dir = output_base_dir / str(q_num)
    question_dir.mkdir(parents=True, exist_ok=True)

    # Prompt: explicitly ask for 1–2 full sentences
    prompt_text = (
        "You are a helpful, concise tutor.\n\n"
        "Answer the question below in 1–2 complete sentences, using clear natural language.\n"
        "Do not answer with a single word or a fragment.\n\n"
        f"Question:\n{q_text}\n"
    )

    # write prompt locally and push to device
    local_prompt_path = script_dir / "prompt.txt"
    local_prompt_path.write_text(prompt_text, encoding="utf-8")
    subprocess.run(
        ["adb", "push", str(local_prompt_path), REMOTE_PROMPT_FILE],
        check=True,
    )

    for model_file in model_files:
        model_path = f"{MODEL_BASE_DIR}/{model_file}"

        # temporary raw output file on host
        local_raw_result = script_dir / "tmp_result_raw.json"

        # per-model output file on host
        output_file_path = question_dir / f"{model_file}.json"

        # build command to run on device
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
            f"--json-schema '{json_schema_str}' "
            f"--file {REMOTE_PROMPT_FILE} "
            f"> {REMOTE_RESULT_FILE}"
        )

        # measure inference time like generate_summary.py
        start_time = time.time()
        result = subprocess.run(
            ["adb", "shell", adb_cmd],
            capture_output=True,
            text=True,
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            # on error, save diagnostics + time
            error_payload = {
                "inference_time_seconds": duration,
                "response": {
                    "error": "inference_failed",
                    "stderr": result.stderr or "",
                },
            }
            output_file_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        # pull raw result to host
        subprocess.run(
            ["adb", "pull", REMOTE_RESULT_FILE, str(local_raw_result)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # read and parse like generate_qna.py
        try:
            raw_text = local_raw_result.read_text(encoding="utf-8")
        except FileNotFoundError:
            error_payload = {
                "inference_time_seconds": duration,
                "response": {
                    "error": "missing_result_file",
                },
            }
            output_file_path.write_text(
                json.dumps(error_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        parsed = extract_json_from_output(raw_text)
        if parsed is None:
            # fall back to saving raw text if parsing fails
            payload = {
                "inference_time_seconds": duration,
                "response": {
                    "raw": raw_text,
                    "parse_warning": "could_not_parse_json",
                },
            }
        else:
            payload = {
                "inference_time_seconds": duration,
                "response": parsed,
            }

        with output_file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # cleanup temp raw file
        if local_raw_result.exists():
            local_raw_result.unlink(missing_ok=True)


def main():
    # load all questions
    try:
        questions = json.loads(questions_file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"questions.json not found at {questions_file_path}", file=sys.stderr)
        sys.exit(1)

    for q in questions:
        run_one_question(q)


if __name__ == "__main__":
    main()



