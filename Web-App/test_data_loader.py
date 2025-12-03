import pandas as pd
from pathlib import Path
from data_loader import load_metrics, load_multiple_metrics, available_models, available_tasks, aggregate_accuracy, latest_accuracy

ROOT = Path(__file__).resolve().parent.parent  # project root
SAMPLE_CSV = ROOT / 'sample_metrics.csv'


def test_loading():
    df = load_metrics(SAMPLE_CSV)
    assert not df.empty
    assert set(['model','task','temperature','memory_used_mb','cpu_percent','accuracy']).issubset(df.columns)


def test_models_tasks():
    df = load_metrics(SAMPLE_CSV)
    models = available_models(df)
    assert 'slm-alpha' in models and 'slm-beta' in models
    tasks_alpha = available_tasks(df, 'slm-alpha')
    assert 'text-generation' in tasks_alpha and 'classification' in tasks_alpha


def test_aggregate_accuracy():
    df = load_metrics(SAMPLE_CSV)
    agg = aggregate_accuracy(df, 'slm-alpha', 'text-generation')
    assert 'accuracy_mean' in agg.columns


def test_latest_accuracy():
    df = load_metrics(SAMPLE_CSV)
    latest = latest_accuracy(df, 'slm-beta', 'classification')
    assert isinstance(latest, float)


def test_multi_file_merge():
    # Split sample metrics into two temp files
    import pandas as pd
    df = load_metrics(SAMPLE_CSV)
    part1 = df[['timestamp','model','task','accuracy']]
    part2 = df[['timestamp','model','task','temperature','memory_used_mb','cpu_percent']]
    tmp1 = ROOT / 'tmp_part1.csv'
    tmp2 = ROOT / 'tmp_part2.csv'
    part1.to_csv(tmp1, index=False)
    part2.to_csv(tmp2, index=False)
    merged = load_multiple_metrics([tmp1, tmp2])
    assert 'accuracy' in merged.columns and 'temperature' in merged.columns
    assert len(merged) == len(df.groupby(['timestamp','model','task'], as_index=False).last())


def test_per_task_split():
    # Simulate splitting tasks into separate files
    base = load_metrics(SAMPLE_CSV)
    tasks = base['task'].unique()
    temp_files = []
    for t in tasks:
        part = base[base['task']==t]
        tmp = ROOT / f'tmp_{t}.csv'
        part.to_csv(tmp, index=False)
        temp_files.append(tmp)
    merged = load_multiple_metrics(temp_files)
    assert set(merged['task'].unique()) == set(tasks)
    # Ensure row count equals original aggregation by unique key
    orig = base.groupby(['timestamp','model','task'], as_index=False).last()
    assert len(merged) == len(orig)

if __name__ == '__main__':
    test_loading()
    test_models_tasks()
    test_aggregate_accuracy()
    test_latest_accuracy()
    print('All tests passed.')
