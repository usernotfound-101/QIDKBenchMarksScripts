# filepath: /home/usernotfound101/final-codes-48_pikachu/Code/Implementation/evaluate_scores_per_question.py
import json
import string
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Optional BERTScore
try:
    from bert_score import score as bert_score
except ImportError:  # pragma: no cover
    bert_score = None


ROOT = Path(__file__).resolve().parent
ANSWERS_PATH = ROOT / "answers.json"
OUTPUTS_DIR = ROOT / "outputs"
CSV_DIR = ROOT / "csv_files_per_model"
BEST_OUT_PATH = ROOT / "best_models_per_question.json"


def load_carbon_scores() -> Dict[str, float]:
    """
    Load all model CSVs, compute COavg per model (mean of CO2_rate_g_per_s or a proxy),
    then compute global COmin/COmax and CO_norm/CarbonScore per model.
    Returns: { model_name (csv filename) : CarbonScore }
    """
    model_co2: Dict[str, float] = {}

    for csv_path in CSV_DIR.glob("*.csv"):
        model_name = csv_path.name  # key used later as "<model>.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        # Try explicit CO2 rate column if present
        co2_cols = [c for c in df.columns if "co2" in c.lower() and "rate" in c.lower()]
        if co2_cols:
            col = co2_cols[0]
            vals = pd.to_numeric(df[col], errors="coerce")
            # ensure Series
            if not isinstance(vals, pd.Series):
                vals = pd.Series([vals])
            vals = vals.dropna()
            if not vals.empty:
                model_co2[model_name] = float(vals.mean())
                continue

        # Fallback proxy using cpu_percent, memory_used_mb, temperature
        cpu = df.get("cpu_percent", pd.Series(index=df.index, dtype=float))
        mem = df.get("memory_used_mb", pd.Series(index=df.index, dtype=float))
        temp = df.get("temperature", pd.Series(index=df.index, dtype=float))

        # convert to numeric Series explicitly
        cpu = pd.to_numeric(cpu, errors="coerce")
        mem = pd.to_numeric(mem, errors="coerce")
        temp = pd.to_numeric(temp, errors="coerce")

        if not isinstance(cpu, pd.Series):
            cpu = pd.Series([cpu])
        if not isinstance(mem, pd.Series):
            mem = pd.Series([mem])
        if not isinstance(temp, pd.Series):
            temp = pd.Series([temp])

        # Fill NaNs with medians or defaults
        cpu = cpu.fillna(cpu.median() if cpu.notna().any() else 0.0)
        mem = mem.fillna(mem.median() if mem.notna().any() else 0.0)
        temp = temp.fillna(temp.median() if temp.notna().any() else 25.0)

        proxy = (cpu / 100.0) * mem * (1.0 + (temp - 25.0).clip(lower=0) / 20.0)
        proxy = proxy.dropna()
        if proxy.empty:
            continue

        model_co2[model_name] = float(proxy.mean())

    if not model_co2:
        return {}

    co_values = list(model_co2.values())
    co_min = min(co_values)
    co_max = max(co_values)

    carbon_scores: Dict[str, float] = {}
    for model_name, co_avg in model_co2.items():
        if co_max == co_min:
            co_norm = 0.0
        else:
            co_norm = (co_avg - co_min) / (co_max - co_min)
        carbon_score = 1.0 - co_norm
        carbon_scores[model_name] = float(carbon_score)

    return carbon_scores


# Normalization helpers for F1 (same style as evaluation_loader)
def _norm_text(t: str) -> str:
    t = t.replace("[answer start]", "").replace("[answer end]", "")
    t = t.lower()
    t = "".join(ch for ch in t if ch not in set(string.punctuation))
    return " ".join(t.split())


def _f1(pred: str, gold: str) -> float:
    pt = _norm_text(pred).split()
    gt = _norm_text(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = set(pt) & set(gt)
    num_same = sum(min(pt.count(tok), gt.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return float(2 * precision * recall / (precision + recall))


def bert_f1_single(pred: str, gold: str) -> float:
    if not bert_score:
        return float("nan")
    try:
        P, R, F1 = bert_score([pred], [gold], lang="en", verbose=False)
        return float(F1[0].item())
    except Exception:
        return float("nan")


def load_gold_answers() -> Dict[int, Dict[str, Any]]:
    """Return mapping { index : {question, answer} } from answers.json."""
    data = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, Any]] = {}
    for item in data.get("faq_list", []):
        idx = item.get("index")
        if isinstance(idx, int):
            out[idx] = {
                "question": item.get("question", "").strip(),
                "answer": item.get("answer", "").strip(),
            }
    return out


def collect_model_answers_for_question(qid: int) -> Dict[str, Dict[str, Any]]:
    """
    For a given question id (directory name under outputs),
    load each model's JSON and return:
    {
      model_filename: {
         'answer': str,
         'inference_time': float
      }
    }
    """
    q_dir = OUTPUTS_DIR / str(qid)
    if not q_dir.exists():
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for path in q_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        inf_time = float(obj.get("inference_time_seconds", float("nan")))
        resp = obj.get("response", {})
        ans_list = resp.get("answers") if isinstance(resp, dict) else None
        if not ans_list or not isinstance(ans_list, list):
            continue
        first = ans_list[0]
        model_answer = first.get("answer", "")
        if not isinstance(model_answer, str):
            model_answer = str(model_answer)
        results[path.name.replace(".json", "")] = {
            "answer": model_answer.strip(),
            "inference_time": inf_time,
        }
    return results


def main():
    carbon_scores = load_carbon_scores()
    gold = load_gold_answers()
    if not gold:
        raise SystemExit("No gold answers loaded from answers.json")

    best_per_question: List[Dict[str, Any]] = []

    # Iterate over question IDs that have outputs and gold answers
    q_ids = sorted(int(p.name) for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    for qid in q_ids:
        if qid not in gold:
            continue
        gold_q = gold[qid]["question"]
        gold_a = gold[qid]["answer"]

        model_answers = collect_model_answers_for_question(qid)
        if not model_answers:
            continue

        best_model_name = None
        best_score = float("-inf")
        best_record: Dict[str, Any] = {}

        for model_name, info in model_answers.items():
            pred = info["answer"]
            inf_t = info["inference_time"]
            if not pred or not isinstance(inf_t, (int, float)) or inf_t <= 0:
                continue

            # Per-question metrics
            f1 = _f1(pred, gold_a)
            bert_f = bert_f1_single(pred, gold_a)
            if bert_f != bert_f:  # NaN check
                bert_f = f1  # fallback to F1 if BERTScore unavailable

            # CarbonScore is precomputed per CSV filename; if missing, treat as neutral 1.0
            carbon = carbon_scores.get(f"{model_name}.csv", 1.0)

            # FinalScore = ((0.6⋅CarbonScore)+(0.25⋅F1)+(0.15⋅BERTScore))/Inference_Time
            numerator = (0.6 * carbon) + (0.25 * f1) + (0.15 * bert_f)
            final_score = numerator / inf_t

            if final_score > best_score:
                best_score = final_score
                best_model_name = model_name
                best_record = {
                    "question_id": qid,
                    "question": gold_q,
                    "gold_answer": gold_a,
                    "best_model": model_name,
                    "model_answer": pred,
                    "metrics": {
                        "F1": f1,
                        "BERTScore_F1": bert_f,
                        "CarbonScore": carbon,
                        "Inference_Time": inf_t,
                        "FinalScore": final_score,
                    },
                }

        if best_model_name is not None:
            best_per_question.append(best_record)

    BEST_OUT_PATH.write_text(
        json.dumps(best_per_question, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote best models per question to {BEST_OUT_PATH}")


if __name__ == "__main__":
    main()
