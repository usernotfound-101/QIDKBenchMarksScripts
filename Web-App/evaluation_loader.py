import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import statistics

try:
    from bert_score import score as bert_score
except ImportError:  # pragma: no cover
    bert_score = None

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    cosine_similarity = None

# Accept multiple possible key names in user JSONs
QA_PRED_KEYS = {
    "answer", "prediction", "pred", "generated_answer", "model_answer",
    "output", "response", "generated_text", "output_text", "model_output"
}
QA_REF_KEYS = {
    "expected", "reference", "gold", "ground_truth", "target",
    "answer_text", "ideal_answer"
}
SUM_PRED_KEYS = {
    "generated_summary", "summary", "generated", "predicted_summary",
    "model_summary", "output_summary", "response", "generated_text",
    "prediction_text", "summary_text", "output"
}
SUM_REF_KEYS = {
    "highlight", "reference", "references", "target", "gold_summary",
    "summary_reference", "highlights", "reference_text", "target_text"
}


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        else:
            # Try JSONL first
            content = f.read()
            f.seek(0)
            lines = [json.loads(line) for line in content.splitlines() if line.strip()]
            if lines:
                return lines
            # Fallback: single JSON object with container list
            f.seek(0)
            obj = json.load(f)
            if isinstance(obj, dict):
                container_keys = ['data','items','records','results','rows','samples','entries','examples']
                for k in container_keys:
                    if k in obj and isinstance(obj[k], list):
                        return obj[k]
                # Special case: parallel arrays
                if isinstance(obj.get('predictions'), list) and isinstance(obj.get('references'), list):
                    preds = obj['predictions']
                    refs = obj['references']
                    n = min(len(preds), len(refs))
                    out = []
                    for i in range(n):
                        out.append({'generated_summary': preds[i], 'highlight': refs[i]})
                    return out
            return []


def _get_first_str(e: Dict[str, Any], keys: set) -> str:
    def coerce_to_text(val: Any) -> str:
        # If nested dict, try common text fields
        if isinstance(val, dict):
            for tkey in ['text','value','content','string','answer','summary','output']:
                if tkey in val and val[tkey] is not None:
                    return coerce_to_text(val[tkey])
            return ""
        if isinstance(val, list):
            parts = [coerce_to_text(x) for x in val]
            parts = [p for p in parts if p]
            return " ".join(parts)
        if val is None:
            return ""
        s = str(val)
        return s.strip()

    for k in keys:
        if k in e and e[k] is not None:
            v = coerce_to_text(e[k])
            if v:
                return v
    return ""


def _extract_answer_tags(text: str) -> str:
    # Extract text between [answer start] and [answer end] if present
    start_tag = "[answer start]"
    end_tag = "[answer end]"
    s = text.find(start_tag)
    e = text.find(end_tag)
    if s != -1 and e != -1 and e > s:
        return text[s + len(start_tag):e].strip()
    return text.strip()


def detect_type(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return 'unknown'
    qa_count = 0
    sum_count = 0
    for e in entries:
        pred_q = _get_first_str(e, QA_PRED_KEYS)
        ref_q = _get_first_str(e, QA_REF_KEYS)
        if pred_q and ref_q:
            qa_count += 1
        pred_s = _get_first_str(e, SUM_PRED_KEYS)
        ref_s = _get_first_str(e, SUM_REF_KEYS)
        if pred_s and ref_s:
            sum_count += 1
    if qa_count and qa_count >= sum_count:
        return 'qa'
    if sum_count:
        return 'summarization'
    return 'unknown'


def compute_qa_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Exact Match / token F1 from f1-em logic
    import string
    def norm(t: str) -> str:
        t = t.replace('[answer start]', '').replace('[answer end]', '')
        t = t.lower()
        t = ''.join(ch for ch in t if ch not in set(string.punctuation))
        return ' '.join(t.split())

    def em(pred: str, gold: str) -> int:
        return int(norm(pred) == norm(gold))

    def f1(pred: str, gold: str) -> float:
        pt = norm(pred).split()
        gt = norm(gold).split()
        if not pt or not gt:
            return float(pt == gt)
        common = set(pt) & set(gt)
        num_same = sum(min(pt.count(tok), gt.count(tok)) for tok in common)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pt)
        recall = num_same / len(gt)
        return 2 * precision * recall / (precision + recall)

    # Normalize keys and extract answers
    preds: List[str] = []
    refs: List[str] = []
    for e in entries:
        pred_raw = _get_first_str(e, QA_PRED_KEYS)
        ref_raw = _get_first_str(e, QA_REF_KEYS)
        if not pred_raw or not ref_raw:
            continue
        preds.append(_extract_answer_tags(pred_raw))
        refs.append(_extract_answer_tags(ref_raw))

    em_scores = []
    f1_scores = []
    for p, r in zip(preds, refs):
        em_scores.append(em(p, r))
        f1_scores.append(f1(p, r))

    def stats(arr):
        return {
            'mean': statistics.mean(arr) if arr else float('nan'),
            'median': statistics.median(arr) if arr else float('nan'),
            'std': statistics.pstdev(arr) if len(arr) > 1 else 0.0,
        }

    results = {
        'exact_match': stats(em_scores),
        'f1': stats(f1_scores),
        'count': len(preds),
    }
    # Optional BERTScore between answer and expected
    per_item: Dict[str, List[float]] = {}
    if bert_score and preds:
        try:
            P, R, F1 = bert_score(preds, refs, lang='en', verbose=False)
            import numpy as np
            def stat_np(a):
                return {
                    'mean': float(a.mean()),
                    'median': float(np.median(a)),
                    'std': float(a.std()),
                }
            Pn, Rn, F1n = P.numpy(), R.numpy(), F1.numpy()
            results['bertscore_precision'] = stat_np(Pn)
            results['bertscore_recall'] = stat_np(Rn)
            results['bertscore_f1'] = stat_np(F1n)
            per_item['bertscore_f1'] = [float(x) for x in F1n]
        except Exception as e:  # pragma: no cover
            results['bertscore_error'] = str(e)
    # Optional cosine similarity for QA
    if SentenceTransformer and preds:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            pred_vecs = model.encode(preds)
            ref_vecs = model.encode(refs)
            import numpy as np
            sims = []
            if cosine_similarity is not None:
                for gv, rv in zip(pred_vecs, ref_vecs):
                    sims.append(float(cosine_similarity([gv], [rv])[0][0]))
            sims_arr = np.array(sims)
            if sims:
                results['cosine_similarity'] = {
                    'mean': float(sims_arr.mean()),
                    'median': float(np.median(sims_arr)),
                    'std': float(sims_arr.std()),
                }
                per_item['cosine_similarity'] = sims
        except Exception as e:  # pragma: no cover
            results['cosine_error'] = str(e)
    if per_item:
        results['per_item'] = per_item
    return results


def compute_summarization_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    refs: List[str] = []
    gens: List[str] = []
    for e in entries:
        ref = _get_first_str(e, SUM_REF_KEYS).strip()
        gen = _get_first_str(e, SUM_PRED_KEYS).strip()
        if ref and gen:
            refs.append(ref)
            gens.append(gen)

    results: Dict[str, Any] = {'count': len(gens)}

    # ROUGE
    per_item: Dict[str, List[float]] = {}
    if rouge_scorer and gens:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rL = [], [], []
        for g, r in zip(gens, refs):
            res = scorer.score(r, g)
            r1.append(res['rouge1'].fmeasure)
            r2.append(res['rouge2'].fmeasure)
            rL.append(res['rougeL'].fmeasure)
        def stat(a):
            return {
                'mean': statistics.mean(a) if a else float('nan'),
                'median': statistics.median(a) if a else float('nan'),
                'std': statistics.pstdev(a) if len(a) > 1 else 0.0,
            }
        results['rouge1_f1'] = stat(r1)
        results['rouge2_f1'] = stat(r2)
        results['rougeL_f1'] = stat(rL)
        per_item['rougeL_f1'] = rL

    # BERTScore
    if bert_score and gens:
        try:
            P, R, F1 = bert_score(gens, refs, lang='en', verbose=False)
            import numpy as np
            Pn, Rn, F1n = P.numpy(), R.numpy(), F1.numpy()
            def stat_np(a):
                return {
                    'mean': float(a.mean()),
                    'median': float(np.median(a)),
                    'std': float(a.std()),
                }
            results['bertscore_precision'] = stat_np(Pn)
            results['bertscore_recall'] = stat_np(Rn)
            results['bertscore_f1'] = stat_np(F1n)
            per_item['bertscore_f1'] = [float(x) for x in F1n]
        except Exception as e:  # pragma: no cover
            results['bertscore_error'] = str(e)

    # Cosine similarity embeddings
    if SentenceTransformer and gens:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            gen_vecs = model.encode(gens)
            ref_vecs = model.encode(refs)
            import numpy as np
            sims = []
            for gv, rv in zip(gen_vecs, ref_vecs):
                if cosine_similarity is not None:
                    sims.append(float(cosine_similarity([gv], [rv])[0][0]))
            sims_arr = np.array(sims)
            if sims:
                results['cosine_similarity'] = {
                    'mean': float(sims_arr.mean()),
                    'median': float(np.median(sims_arr)),
                    'std': float(sims_arr.std()),
                }
                per_item['cosine_similarity'] = sims
        except Exception as e:  # pragma: no cover
            results['cosine_error'] = str(e)
    if not gens or not refs:
        results['missing_reason'] = 'No (prediction, reference) pairs found with expected keys.'
    if per_item:
        results['per_item'] = per_item
    return results


def load_evaluation_file(path: Path) -> Dict[str, Any]:
    entries = _load_json_or_jsonl(path)
    data_type = detect_type(entries)
    if data_type == 'qa':
        metrics = compute_qa_metrics(entries)
    elif data_type == 'summarization':
        metrics = compute_summarization_metrics(entries)
    else:
        metrics = {'count': 0}
    return {'type': data_type, 'metrics': metrics, 'source': str(path)}


def merge_metric_sets(sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Combine counts and list individual sources
    return {
        'sources': [s['source'] for s in sets],
        'details': sets,
    }
