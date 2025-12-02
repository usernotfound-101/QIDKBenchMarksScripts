import pandas as pd
from pathlib import Path
from typing import List, Optional, Sequence

MANDATORY_BASE_COLUMNS = ["timestamp", "model", "task"]
OPTIONAL_METRIC_COLUMNS = ["temperature", "memory_used_mb", "cpu_percent", "accuracy"]
ALL_COLUMNS = MANDATORY_BASE_COLUMNS + OPTIONAL_METRIC_COLUMNS


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_base = [c for c in MANDATORY_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV missing mandatory columns: {missing_base}")
    present_numeric = [c for c in OPTIONAL_METRIC_COLUMNS if c in df.columns]
    for col in present_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    return df


def load_multiple_metrics(paths: Sequence[Path]) -> pd.DataFrame:
    """Load and merge multiple CSV files that may contain subsets of the required columns.

    Rules:
    - All files must have timestamp, model, task.
    - Metric columns (temperature, memory_used_mb, cpu_percent, accuracy) can appear in any file.
    - Rows are concatenated; if multiple rows share same timestamp+model+task, later files override metric values (last-write-wins merge).
    - Returns a DataFrame with full REQUIRED_COLUMNS (may contain NaN if metric never provided for a row).
    """
    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            continue
        raw = pd.read_csv(p)
        basic_missing = [c for c in ["timestamp","model","task"] if c not in raw.columns]
        if basic_missing:
            raise ValueError(f"File {p} missing mandatory columns: {basic_missing}")
        # Keep only known columns
        allowed_cols = [c for c in raw.columns if c in ALL_COLUMNS]
        raw = raw[allowed_cols]
        raw['timestamp'] = pd.to_datetime(raw['timestamp'], errors='coerce')
        raw = raw.dropna(subset=['timestamp'])
        frames.append(raw)

    if not frames:
        return pd.DataFrame(columns=ALL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    # Sort and deduplicate by timestamp-model-task keeping last occurrence
    combined = combined.sort_values('timestamp')
    combined = combined.groupby(['timestamp','model','task'], as_index=False).last()
    # Ensure all required columns exist
    for col in ALL_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    present_numeric = [c for c in OPTIONAL_METRIC_COLUMNS if c in combined.columns]
    for col in present_numeric:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    return combined.sort_values('timestamp')


def available_models(df: pd.DataFrame) -> List[str]:
    return sorted(df['model'].unique())


def available_tasks(df: pd.DataFrame, model: Optional[str] = None) -> List[str]:
    subset = df if model is None else df[df['model'] == model]
    return sorted(subset['task'].unique())


def aggregate_accuracy(df: pd.DataFrame, model: str, task: Optional[str] = None) -> pd.DataFrame:
    subset = df[df['model'] == model]
    if task:
        subset = subset[subset['task'] == task]
    if 'accuracy' not in subset.columns:
        return pd.DataFrame(columns=['timestamp','accuracy_mean'])
    return subset.groupby(pd.Grouper(key='timestamp', freq='h'))['accuracy'].mean().reset_index(name='accuracy_mean')


def latest_accuracy(df: pd.DataFrame, model: str, task: Optional[str] = None) -> float:
    subset = df[df['model'] == model]
    if task:
        subset = subset[subset['task'] == task]
    if subset.empty or 'accuracy' not in subset.columns:
        return float('nan')
    return subset.sort_values('timestamp').iloc[-1].get('accuracy', float('nan'))
