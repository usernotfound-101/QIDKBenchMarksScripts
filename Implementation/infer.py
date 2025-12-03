# filepath: /home/usernotfound101/final-codes-48_pikachu/Code/Implementation/infer.py
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fastembed import TextEmbedding

# Paths
ROOT = Path(__file__).resolve().parent
BEST_PATH = ROOT / "best_models_per_question.json"

# Device paths
PHONE_TMP_DIR = "/data/local/tmp"
LLAMA_BIN_DIR = os.path.join(PHONE_TMP_DIR, "llama/build-clblast/bin")
LLAMA_CLI_PATH = os.path.join(LLAMA_BIN_DIR, "llama-cli")
MODEL_BASE_DIR = os.path.join(PHONE_TMP_DIR, "gguf")
REMOTE_PROMPT_FILE = os.path.join(PHONE_TMP_DIR, "prompt_infer.txt")
REMOTE_RESULT_FILE = os.path.join(PHONE_TMP_DIR, "result_infer.json")

# JSON schema (same as run_each_model, but used only for structure)
JSON_SCHEMA = {
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
JSON_SCHEMA_STR = json.dumps(JSON_SCHEMA)


@dataclass
class QAEntry:
    question_id: int
    question: str
    best_model: str


def load_best_models() -> List[QAEntry]:
    data = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    entries: List[QAEntry] = []
    for item in data:
        entries.append(
            QAEntry(
                question_id=int(item["question_id"]),
                question=item["question"],
                best_model=item["best_model"],  # e.g. "qwen3-4b-instruct-2507-q4km.gguf"
            )
        )
    return entries


def build_index(entries: List[QAEntry]):
    """
    Build an embedding index over stored questions using fastembed.
    Returns: (embedder, questions_list, question_embeddings[np.ndarray])
    """
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    questions = [e.question for e in entries]
    # fastembed returns a generator; materialize to array
    embs = list(embedder.embed(questions))
    embeddings = np.vstack(embs)  # shape: [N, D]
    # L2-normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    return embedder, questions, embeddings


def find_most_similar(
    query: str, embedder: TextEmbedding, questions: List[str], embeddings: np.ndarray
) -> Tuple[int, float]:
    q_emb = np.array(list(embedder.embed([query]))[0], dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    # cosine similarity = dot product since both are normalized
    scores = embeddings @ q_emb  # shape: [N]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    return best_idx, best_score


def extract_json_from_output(content: str):
    """Same robust JSON extraction as in run_each_model.py."""
    try:
        return json.loads(content)
    except Exception:
        pass

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

    import re as _re

    m = _re.search(r"\{.*\}", content, flags=_re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def run_model_on_phone(model_file: str, user_query: str) -> str:
    """
    Run the chosen model on device with a prompt built from user_query.
    Returns the answer text (first item in answers array) or raw fallback.
    """
    prompt_text = (
        "You are a helpful, concise tutor.\n\n"
        "Answer the question below in 1–2 complete sentences, using clear natural language.\n"
        "Do not answer with a single word or a fragment.\n\n"
        f"Question:\n{user_query}\n"
    )

    # Write prompt locally
    local_prompt_path = ROOT / "prompt_infer.txt"
    local_prompt_path.write_text(prompt_text, encoding="utf-8")

    # Push prompt to device
    subprocess.run(
        ["adb", "push", str(local_prompt_path), REMOTE_PROMPT_FILE],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    model_path = f"{MODEL_BASE_DIR}/{model_file}"

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
        f"--json-schema '{JSON_SCHEMA_STR}' "
        f"--file {REMOTE_PROMPT_FILE} "
        f"> {REMOTE_RESULT_FILE}"
    )

    start = time.time()
    result = subprocess.run(
        ["adb", "shell", adb_cmd],
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    if result.returncode != 0:
        return f"[ERROR] Inference failed: {result.stderr.strip()}"

    # Pull result
    local_result_path = ROOT / "result_infer.json"
    subprocess.run(
        ["adb", "pull", REMOTE_RESULT_FILE, str(local_result_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    raw = local_result_path.read_text(encoding="utf-8")
    parsed = extract_json_from_output(raw)

    if not parsed or not isinstance(parsed, dict):
        return f"(Inference time: {duration:.2f}s)\n[WARN] Could not parse JSON. Raw output:\n{raw}"

    resp = parsed.get("answers")
    if not isinstance(resp, list) or not resp:
        return f"(Inference time: {duration:.2f}s)\n[WARN] No answers array in JSON. Raw output:\n{raw}"

    first = resp[0]
    ans = first.get("answer", "")
    if not isinstance(ans, str):
        ans = str(ans)

    return f"(Inference time: {duration:.2f}s)\n{ans.strip()}"


def main():
    if not BEST_PATH.exists():
        raise SystemExit(f"best_models_per_question.json not found at {BEST_PATH}")

    entries = load_best_models()
    if not entries:
        raise SystemExit("No entries found in best_models_per_question.json")

    print("Building semantic index over stored questions (fastembed)...")
    embedder, questions, embeddings = build_index(entries)
    print(f"Indexed {len(questions)} questions.\n")

    while True:
        try:
            user_q = input("Enter your question (or 'quit' to exit): ").strip()
        except EOFError:
            break
        if not user_q:
            continue
        if user_q.lower() in {"q", "quit", "exit"}:
            break

        idx, score = find_most_similar(user_q, embedder, questions, embeddings)
        matched = entries[idx]
        print(f"\nMost similar stored question (cosine={score:.3f}):")
        print(f"  QID {matched.question_id}: {matched.question}")
        print(f"  Using model: {matched.best_model}\n")

        answer = run_model_on_phone(matched.best_model, user_q)
        print("Answer:")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
