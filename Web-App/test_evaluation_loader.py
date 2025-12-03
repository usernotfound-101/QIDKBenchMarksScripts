import json
from pathlib import Path
from evaluation_loader import _load_json_or_jsonl, detect_type, compute_qa_metrics, compute_summarization_metrics, load_evaluation_file

# Simple synthetic QA JSONL
qa_jsonl_path = Path('/tmp/test_qa.jsonl')
qa_entries = [
    {"question": "Q1", "answer": "Paris", "expected": "Paris"},
    {"question": "Q2", "answer": "London", "expected": "Berlin"},
]
with qa_jsonl_path.open('w', encoding='utf-8') as f:
    for e in qa_entries:
        f.write(json.dumps(e) + '\n')

# Synthetic summarization JSON
summ_json_path = Path('/tmp/test_sum.json')
summ_entries = [
    {"generated_summary": "Cats are nice.", "highlight": "Cats are nice and fluffy."},
    {"generated_summary": "Dogs are friendly.", "highlight": "Dogs are very friendly pets."},
]
with summ_json_path.open('w', encoding='utf-8') as f:
    json.dump(summ_entries, f)

# Load & detect
qa_loaded = _load_json_or_jsonl(qa_jsonl_path)
assert len(qa_loaded) == 2, 'QA file load failed'
assert detect_type(qa_loaded) == 'qa', 'QA type detection failed'

sum_loaded = _load_json_or_jsonl(summ_json_path)
assert len(sum_loaded) == 2, 'Summ file load failed'
assert detect_type(sum_loaded) == 'summarization', 'Summ type detection failed'

qa_metrics = compute_qa_metrics(qa_loaded)
assert qa_metrics['count'] == 2, 'QA metrics count mismatch'
assert 'exact_match' in qa_metrics and 'f1' in qa_metrics, 'QA metrics keys missing'
# BERTScore F1 is optional; if present validate range
bert_f1_mean = qa_metrics.get('bertscore_f1_mean') or qa_metrics.get('metrics', {}).get('bertscore_f1_mean')
if bert_f1_mean is not None:
    assert isinstance(bert_f1_mean, float)
    assert 0.0 <= bert_f1_mean <= 1.0

sum_metrics = compute_summarization_metrics(sum_loaded)
assert sum_metrics['count'] == 2, 'Summ metrics count mismatch'
# ROUGE / BERTScore / cosine may be skipped depending on deps; ensure no crash and structure present

qa_wrap = load_evaluation_file(qa_jsonl_path)
assert qa_wrap['type'] == 'qa', 'Wrapped QA type mismatch'
assert qa_wrap['metrics']['count'] == 2, 'Wrapped QA count mismatch'

sum_wrap = load_evaluation_file(summ_json_path)
assert sum_wrap['type'] == 'summarization', 'Wrapped summarization type mismatch'
assert sum_wrap['metrics']['count'] == 2, 'Wrapped summarization count mismatch'

print('Evaluation loader tests passed.')
