#!/usr/bin/env python3
"""Visualization toolkit for RAG benchmark runs with per-stage carbon metrics.

This variant focuses on ``*-ragres2`` folders that include the updated
``rag_benchmark.py`` outputs (per-question retrieval/generation carbon). It loads
summary statistics plus parsed outputs, aggregates the additional metrics, and
renders insight-driven Plotly charts for quick comparisons between models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class ModelReport:
    label: str
    retrieval_latency_ms: Optional[float]
    load_ms: Optional[float]
    prompt_ms: Optional[float]
    eval_ms: Optional[float]
    throughput_eval_tps: Optional[float]
    eval_tokens_avg: Optional[float]
    memory_mb: Optional[float]
    semantic_similarity: Optional[float]
    f1: Optional[float]
    retrieval_carbon_avg: Optional[float]
    generation_carbon_avg: Optional[float]
    question_count: int
    answer_chars_avg: Optional[float]
    avg_graphs_reused: Optional[float]
    avg_retrieved_chunks: Optional[float]
    avg_top_similarity: Optional[float]
    avg_retrieval_carbon_per_chunk: Optional[float]

    @property
    def blended_accuracy(self) -> Optional[float]:
        if self.f1 is None or self.semantic_similarity is None:
            return None
        return 0.5 * (self.f1 + self.semantic_similarity)

    @property
    def generation_latency_ms(self) -> Optional[float]:
        components = [self.load_ms, self.prompt_ms, self.eval_ms]
        if any(component is None for component in components):
            return None
        return sum(component for component in components if component is not None)

    @property
    def total_latency_ms(self) -> Optional[float]:
        generation = self.generation_latency_ms
        if self.retrieval_latency_ms is None or generation is None:
            return None
        return self.retrieval_latency_ms + generation

    @property
    def carbon_ratio(self) -> Optional[float]:
        if not self.generation_carbon_avg or not self.retrieval_carbon_avg:
            return None
        if self.generation_carbon_avg == 0:
            return None
        return self.retrieval_carbon_avg / self.generation_carbon_avg

    @property
    def total_carbon_per_question(self) -> Optional[float]:
        if self.retrieval_carbon_avg is None or self.generation_carbon_avg is None:
            return None
        return self.retrieval_carbon_avg + self.generation_carbon_avg

    @property
    def generation_carbon_per_100_tokens(self) -> Optional[float]:
        if self.generation_carbon_avg is None or not self.eval_tokens_avg:
            return None
        if self.eval_tokens_avg == 0:
            return None
        return (self.generation_carbon_avg / self.eval_tokens_avg) * 100.0


@dataclass
class ParsedAggregate:
    retrieval_carbon_avg: Optional[float]
    generation_carbon_avg: Optional[float]
    question_count: int
    avg_graphs_reused: Optional[float]
    avg_retrieved_chunks: Optional[float]
    avg_top_similarity: Optional[float]
    avg_retrieval_carbon_per_chunk: Optional[float]


@dataclass
class ChartSection:
    title: str
    description: str
    figure: go.Figure
    div_id: str


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG dashboard (ragres2)")
    parser.add_argument(
        "--rag-root",
        default=".",
        help="Root containing *-ragres2 folders",
    )
    parser.add_argument(
        "--output",
        default="rag_dashboard_2.html",
        help="Path to the generated HTML dashboard",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(container: Dict[str, object], key: str) -> Optional[float]:
    value = container.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metric_avg(metrics_root: Dict[str, object], metric_name: str) -> Optional[float]:
    bucket = metrics_root.get(metric_name)
    if isinstance(bucket, dict):
        avg = bucket.get("avg")
        if isinstance(avg, (int, float)):
            return float(avg)
    return None


def _find_parsed_file(model_dir: Path) -> Optional[Path]:
    matches = sorted(model_dir.glob("parsed_outputs_*.json"))
    return matches[0] if matches else None


def _aggregate_parsed_metrics(parsed_path: Path) -> ParsedAggregate:
    data = _load_json(parsed_path)
    if not isinstance(data, list) or not data:
        return ParsedAggregate(None, None, 0, None, None, None, None)
    retrieval_values: List[float] = []
    generation_values: List[float] = []
    graphs_reused_vals: List[float] = []
    retrieved_chunks_vals: List[float] = []
    top_similarity_vals: List[float] = []
    carbon_per_chunk_vals: List[float] = []
    for entry in data:
        rc = entry.get("retrieval_carbon_kg")
        gc = entry.get("generation_carbon_kg") or entry.get("carbon_emissions_kg")
        if isinstance(rc, (int, float)):
            retrieval_values.append(float(rc))
        if isinstance(gc, (int, float)):
            generation_values.append(float(gc))
        graphs = entry.get("graphs_reused")
        if isinstance(graphs, (int, float)):
            graphs_reused_vals.append(float(graphs))
        retrieved_chunks = entry.get("retrieved_chunks")
        if isinstance(retrieved_chunks, list):
            retrieved_chunks_vals.append(float(len(retrieved_chunks)))
            if retrieved_chunks:
                first = retrieved_chunks[0]
                sim = first.get("similarity") if isinstance(first, dict) else None
                if isinstance(sim, (int, float)):
                    top_similarity_vals.append(float(sim))
        if isinstance(rc, (int, float)) and isinstance(retrieved_chunks, list) and retrieved_chunks:
            carbon_per_chunk_vals.append(float(rc) / len(retrieved_chunks))
    retrieval_avg = mean(retrieval_values) if retrieval_values else None
    generation_avg = mean(generation_values) if generation_values else None
    avg_graphs_reused = mean(graphs_reused_vals) if graphs_reused_vals else None
    avg_retrieved_chunks = mean(retrieved_chunks_vals) if retrieved_chunks_vals else None
    avg_top_similarity = mean(top_similarity_vals) if top_similarity_vals else None
    avg_carbon_per_chunk = mean(carbon_per_chunk_vals) if carbon_per_chunk_vals else None
    return ParsedAggregate(
        retrieval_carbon_avg=retrieval_avg,
        generation_carbon_avg=generation_avg,
        question_count=len(data),
        avg_graphs_reused=avg_graphs_reused,
        avg_retrieved_chunks=avg_retrieved_chunks,
        avg_top_similarity=avg_top_similarity,
        avg_retrieval_carbon_per_chunk=avg_carbon_per_chunk,
    )


def collect_reports(root: Path) -> List[ModelReport]:
    summary_paths = sorted(root.glob("*ragres2*/*/summary.json"))
    reports: List[ModelReport] = []
    for summary_path in summary_paths:
        model_dir = summary_path.parent
        summary_metrics_path = model_dir / "summary_metrics.json"
        if not summary_metrics_path.exists():
            continue
        parsed_path = _find_parsed_file(model_dir)
        if not parsed_path:
            continue

        summary_payload = _load_json(summary_path)
        metrics_payload = _load_json(summary_metrics_path)
        metrics_root = metrics_payload.get("metrics") or {}
        retrieval_block = summary_payload.get("retrieval_latency_ms") or {}
        evaluation_block = summary_payload.get("evaluation") or {}

        parsed_agg = _aggregate_parsed_metrics(parsed_path)
        if parsed_agg.question_count == 0:
            continue

        report = ModelReport(
            label=model_dir.parent.name,
            retrieval_latency_ms=_safe_float(retrieval_block, "avg"),
            load_ms=_metric_avg(metrics_root, "load_time_ms"),
            prompt_ms=_metric_avg(metrics_root, "prompt_eval_time_ms"),
            eval_ms=_metric_avg(metrics_root, "eval_time_ms"),
            throughput_eval_tps=_metric_avg(metrics_root, "eval_tokens_per_sec"),
            eval_tokens_avg=_metric_avg(metrics_root, "eval_tokens"),
            memory_mb=_metric_avg(metrics_root, "memory_usage_mb"),
            semantic_similarity=_safe_float(summary_payload, "semantic_similarity_avg"),
            f1=_safe_float(evaluation_block, "f1"),
            retrieval_carbon_avg=parsed_agg.retrieval_carbon_avg,
            generation_carbon_avg=parsed_agg.generation_carbon_avg,
            question_count=parsed_agg.question_count,
            answer_chars_avg=_metric_avg(metrics_root, "answer_length_chars"),
            avg_graphs_reused=parsed_agg.avg_graphs_reused,
            avg_retrieved_chunks=parsed_agg.avg_retrieved_chunks,
            avg_top_similarity=parsed_agg.avg_top_similarity,
            avg_retrieval_carbon_per_chunk=parsed_agg.avg_retrieval_carbon_per_chunk,
        )
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_carbon_split_chart(reports: List[ModelReport]) -> go.Figure:
    labels = [r.label for r in reports]
    retrieval = [r.retrieval_carbon_avg for r in reports]
    generation = [r.generation_carbon_avg for r in reports]
    fig = go.Figure()
    fig.add_bar(name="Retrieval", x=labels, y=retrieval, marker_color="#06b6d4")
    fig.add_bar(name="Generation", x=labels, y=generation, marker_color="#f97316")
    fig.update_layout(
        title="Per-Question Carbon Split",
        xaxis_title="Model",
        yaxis_title="kg CO₂e",
        barmode="stack",
        template="plotly_white",
    )
    return fig


def build_accuracy_vs_carbon_chart(reports: List[ModelReport]) -> go.Figure:
    labels: List[str] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    for report in reports:
        if report.blended_accuracy is None or report.generation_carbon_avg is None:
            continue
        labels.append(report.label)
        x_vals.append(report.generation_carbon_avg)
        y_vals.append(report.blended_accuracy)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                text=labels,
                mode="markers+text",
                textposition="top center",
                marker=dict(size=14, color="#2563eb", line=dict(width=1, color="#1e1b4b")),
            )
        ]
    )
    fig.update_layout(
        title="Blended Accuracy vs Generation Carbon",
        xaxis_title="Generation Carbon per Question (kg)",
        yaxis_title="(F1 + Semantic) / 2",
        template="plotly_white",
    )
    return fig


def build_latency_vs_carbon_bubble(reports: List[ModelReport]) -> go.Figure:
    x_vals: List[float] = []
    y_vals: List[float] = []
    sizes: List[float] = []
    labels: List[str] = []
    for report in reports:
        if report.generation_latency_ms is None or report.generation_carbon_avg is None or report.retrieval_latency_ms is None:
            continue
        labels.append(report.label)
        x_vals.append(report.generation_latency_ms)
        y_vals.append(report.generation_carbon_avg)
        sizes.append(max(report.retrieval_latency_ms, 0.1) * 80.0)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    sizemode="diameter",
                    sizemin=12,
                    color="#22c55e",
                    opacity=0.7,
                ),
            )
        ]
    )
    fig.update_layout(
        title="Generation Latency vs Carbon (bubble size = retrieval latency)",
        xaxis_title="Generation Latency (ms)",
        yaxis_title="Generation Carbon per Question (kg)",
        template="plotly_white",
    )
    return fig


def build_retrieval_latency_vs_carbon(reports: List[ModelReport]) -> go.Figure:
    labels = []
    latencies = []
    carbons = []
    for report in reports:
        if report.retrieval_latency_ms is None or report.retrieval_carbon_avg is None:
            continue
        labels.append(report.label)
        latencies.append(report.retrieval_latency_ms)
        carbons.append(report.retrieval_carbon_avg)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=latencies,
                y=carbons,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=12, color="#0ea5e9"),
            )
        ]
    )
    fig.update_layout(
        title="Retrieval Latency vs Retrieval Carbon",
        xaxis_title="Retrieval Latency (ms)",
        yaxis_title="Retrieval Carbon per Question (kg)",
        template="plotly_white",
    )
    return fig


def build_throughput_vs_carbon(reports: List[ModelReport]) -> go.Figure:
    labels = []
    throughput = []
    carbon_intensity = []
    for report in reports:
        intensity = report.generation_carbon_per_100_tokens
        if report.throughput_eval_tps is None or intensity is None:
            continue
        labels.append(report.label)
        throughput.append(report.throughput_eval_tps)
        carbon_intensity.append(intensity)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=throughput,
                y=carbon_intensity,
                mode="markers+text",
                text=labels,
                marker=dict(size=14, color="#a855f7"),
            )
        ]
    )
    fig.update_layout(
        title="Eval Throughput vs Carbon Intensity",
        xaxis_title="Eval Throughput (tokens/sec)",
        yaxis_title="Generation Carbon per 100 Tokens (kg)",
        template="plotly_white",
    )
    return fig


def build_carbon_ratio_chart(reports: List[ModelReport]) -> go.Figure:
    labels = []
    ratios = []
    for report in reports:
        ratio = report.carbon_ratio
        if ratio is None:
            continue
        labels.append(report.label)
        ratios.append(ratio)
    fig = go.Figure(
        data=[go.Bar(x=labels, y=ratios, marker_color="#facc15")]
    )
    fig.update_layout(
        title="Retrieval-to-Generation Carbon Ratio",
        xaxis_title="Model",
        yaxis_title="Retrieval Carbon / Generation Carbon",
        template="plotly_white",
    )
    return fig


def build_answer_length_vs_carbon_chart(reports: List[ModelReport]) -> go.Figure:
    points: List[Tuple[str, float, float]] = []
    for report in reports:
        if report.answer_chars_avg is None or report.generation_carbon_avg is None:
            continue
        points.append((report.label, report.answer_chars_avg, report.generation_carbon_avg))
    points.sort(key=lambda item: item[1])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[item[1] for item in points],
                y=[item[2] for item in points],
                text=[item[0] for item in points],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=14, color="#f472b6"),
                hovertemplate="%{text}<br>Answer chars: %{x:.0f}<br>Gen carbon: %{y:.2e} kg<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Answer Length vs Generation Carbon",
        xaxis_title="Average answer length (chars)",
        yaxis_title="Generation carbon per question (kg)",
        template="plotly_white",
    )
    return fig


def build_retrieval_depth_vs_accuracy_chart(reports: List[ModelReport]) -> go.Figure:
    points: List[Tuple[str, float, float]] = []
    for report in reports:
        depth = report.avg_retrieved_chunks
        accuracy = report.blended_accuracy
        if depth is None or accuracy is None:
            continue
        points.append((report.label, depth, accuracy))
    points.sort(key=lambda item: item[1])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[item[1] for item in points],
                y=[item[2] for item in points],
                text=[item[0] for item in points],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=14, color="#22c55e"),
                hovertemplate="%{text}<br>Avg retrieved chunks: %{x:.1f}<br>Blended accuracy: %{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Retrieval Depth vs Blended Accuracy",
        xaxis_title="Avg retrieved chunks per question",
        yaxis_title="(F1 + Semantic) / 2",
        template="plotly_white",
    )
    return fig


def build_carbon_per_chunk_vs_similarity_chart(reports: List[ModelReport]) -> go.Figure:
    points: List[Tuple[str, float, float]] = []
    for report in reports:
        carbon_per_chunk = report.avg_retrieval_carbon_per_chunk
        similarity = report.avg_top_similarity
        if carbon_per_chunk is None or similarity is None:
            continue
        points.append((report.label, carbon_per_chunk, similarity))
    points.sort(key=lambda item: item[1])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[item[1] for item in points],
                y=[item[2] for item in points],
                text=[item[0] for item in points],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=14, color="#0ea5e9"),
                hovertemplate="%{text}<br>Carbon per chunk: %{x:.2e} kg<br>Top chunk similarity: %{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Carbon per Retrieved Chunk vs Similarity",
        xaxis_title="Retrieval carbon per chunk (kg)",
        yaxis_title="Top chunk similarity",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart description helpers
# ---------------------------------------------------------------------------


def _format_value(value: Optional[float], fmt: str) -> Optional[str]:
    if value is None:
        return None
    return format(value, fmt)


def _describe_carbon_split(reports: List[ModelReport]) -> str:
    lowest = _best_by(reports, key=lambda r: r.total_carbon_per_question, reverse=False)
    highest = _best_by(reports, key=lambda r: r.total_carbon_per_question)
    low_val = _format_value(lowest.total_carbon_per_question if lowest else None, ".2e")
    high_val = _format_value(highest.total_carbon_per_question if highest else None, ".2e")
    if lowest and highest and low_val and high_val and lowest.label != highest.label:
        return (
            f"Stacked bars show retrieval vs generation carbon per question. "
            f"{lowest.label} is most frugal at {low_val} kg/query, while {highest.label} sits highest at {high_val} kg/query."
        )
    if lowest and low_val:
        return f"Stacked bars show retrieval vs generation carbon per question. {lowest.label} leads with {low_val} kg/query."
    return "Stacked bars contrast retrieval and generation carbon per question for each run."


def _describe_accuracy_vs_carbon(reports: List[ModelReport]) -> str:
    leader = _best_by(reports, key=lambda r: r.blended_accuracy)
    carbon_saver = _best_by(reports, key=lambda r: r.generation_carbon_avg, reverse=False)
    leader_val = _format_value(leader.blended_accuracy if leader else None, ".3f")
    saver_val = _format_value(carbon_saver.generation_carbon_avg if carbon_saver else None, ".2e")
    if leader and carbon_saver and leader_val and saver_val:
        return (
            f"Scatter illustrates the accuracy/carbon trade-off. {leader.label} tops blended accuracy ({leader_val}), "
            f"while {carbon_saver.label} keeps generation carbon lowest at {saver_val} kg/question."
        )
    if leader and leader_val:
        return f"Markers let you gauge if higher accuracy ({leader.label} at {leader_val}) costs extra carbon."
    return "Markers show how generation carbon per question tracks against blended accuracy."


def _describe_latency_vs_carbon(reports: List[ModelReport]) -> str:
    fastest = _best_by(reports, key=lambda r: r.generation_latency_ms, reverse=False)
    greenest = _best_by(reports, key=lambda r: r.generation_carbon_avg, reverse=False)
    fast_val = _format_value(fastest.generation_latency_ms if fastest else None, ".0f")
    green_val = _format_value(greenest.generation_carbon_avg if greenest else None, ".2e")
    if fastest and fast_val and greenest and green_val:
        return (
            f"Bubble size encodes retrieval latency. {fastest.label} generates answers fastest (~{fast_val} ms), "
            f"while {greenest.label} keeps generation carbon lowest (~{green_val} kg/question)."
        )
    if fastest and fast_val:
        return f"Bubble chart highlights latency-carbon balance, with {fastest.label} around {fast_val} ms generation latency."
    return "Bubble chart maps generation latency (x) vs carbon (y) with bubble size indicating retrieval latency."


def _describe_retrieval_latency_vs_carbon(reports: List[ModelReport]) -> str:
    fastest = _best_by(reports, key=lambda r: r.retrieval_latency_ms, reverse=False)
    cleanest = _best_by(reports, key=lambda r: r.retrieval_carbon_avg, reverse=False)
    fast_val = _format_value(fastest.retrieval_latency_ms if fastest else None, ".0f")
    clean_val = _format_value(cleanest.retrieval_carbon_avg if cleanest else None, ".2e")
    if fastest and fast_val and cleanest and clean_val:
        return (
            f"Dots show retrieval efficiency. {fastest.label} fetches passages quickest (~{fast_val} ms), "
            f"and {cleanest.label} keeps retrieval carbon lowest (~{clean_val} kg/question)."
        )
    if fastest and fast_val:
        return f"Dots show retrieval latency vs carbon, with {fastest.label} near {fast_val} ms."
    return "Dots compare retrieval latency and carbon per query."


def _describe_throughput_vs_carbon(reports: List[ModelReport]) -> str:
    fastest = _best_by(reports, key=lambda r: r.throughput_eval_tps)
    cleanest = _best_by(reports, key=lambda r: r.generation_carbon_per_100_tokens, reverse=False)
    fast_val = _format_value(fastest.throughput_eval_tps if fastest else None, ".1f")
    clean_val = _format_value(cleanest.generation_carbon_per_100_tokens if cleanest else None, ".2e")
    if fastest and fast_val and cleanest and clean_val:
        return (
            f"Higher throughput does not always mean dirtier tokens. {fastest.label} pushes ~{fast_val} tok/s, "
            f"while {cleanest.label} minimizes carbon intensity at {clean_val} kg/100 tok."
        )
    if fastest and fast_val:
        return f"Chart contrasts evaluation throughput and carbon per 100 tokens, with {fastest.label} peaking near {fast_val} tok/s."
    return "Chart contrasts evaluation throughput and carbon per 100 tokens."


def _describe_carbon_ratio(reports: List[ModelReport]) -> str:
    best = _best_by(reports, key=lambda r: r.carbon_ratio, reverse=False)
    worst = _best_by(reports, key=lambda r: r.carbon_ratio)
    best_val = _format_value(best.carbon_ratio if best else None, ".2f")
    worst_val = _format_value(worst.carbon_ratio if worst else None, ".2f")
    if best and worst and best_val and worst_val and best.label != worst.label:
        return (
            f"Bars highlight how much carbon retrieval adds relative to generation. "
            f"{best.label} keeps the ratio lowest ({best_val}), whereas {worst.label} tops the list at {worst_val}."
        )
    if best and best_val:
        return f"Bars highlight retrieval vs generation carbon ratio, led by {best.label} at {best_val}."
    return "Bars highlight how much carbon retrieval adds relative to generation."


def _describe_answer_length_vs_carbon(reports: List[ModelReport]) -> str:
    longest = _best_by(reports, key=lambda r: r.answer_chars_avg)
    greenest = _best_by(reports, key=lambda r: r.generation_carbon_avg, reverse=False)
    longest_len = _format_value(longest.answer_chars_avg if longest else None, ".0f")
    green_val = _format_value(greenest.generation_carbon_avg if greenest else None, ".2e")
    if longest and greenest and longest_len and green_val:
        if longest.label == greenest.label:
            return (
                f"Plot reveals when long answers burn carbon. {longest.label} writes ~{longest_len} chars yet still leads on low carbon ({green_val} kg/question), "
                f"showing verbosity alone doesn't force emissions."
            )
        return (
            f"Plot shows whether longer answers cost carbon. {longest.label} stretches responses (~{longest_len} chars) while {greenest.label} keeps carbon lowest ({green_val} kg/question)."
        )
    return "Plot shows whether writing longer answers noticeably changes generation carbon."


def _describe_retrieval_depth_vs_accuracy(reports: List[ModelReport]) -> str:
    shallow = _best_by(reports, key=lambda r: r.avg_retrieved_chunks, reverse=False)
    deep = _best_by(reports, key=lambda r: r.avg_retrieved_chunks)
    sharp = _best_by(reports, key=lambda r: r.blended_accuracy)
    shallow_val = _format_value(shallow.avg_retrieved_chunks if shallow else None, ".1f")
    deep_val = _format_value(deep.avg_retrieved_chunks if deep else None, ".1f")
    sharp_val = _format_value(sharp.blended_accuracy if sharp else None, ".3f")
    if shallow and deep and sharp and shallow_val and deep_val and sharp_val:
        return (
            f"Helps decide retrieval depth: {shallow.label} keeps it lean (~{shallow_val} chunks) while {deep.label} trawls {deep_val}+ chunks. {sharp.label} still tops accuracy ({sharp_val}), so editors can see if extra fetches truly help."
        )
    return "Shows whether fetching more chunks per query actually correlates with blended accuracy."


def _describe_carbon_per_chunk_vs_similarity(reports: List[ModelReport]) -> str:
    clean = _best_by(reports, key=lambda r: r.avg_retrieval_carbon_per_chunk, reverse=False)
    sticky = _best_by(reports, key=lambda r: r.avg_top_similarity)
    clean_val = _format_value(clean.avg_retrieval_carbon_per_chunk if clean else None, ".2e")
    sticky_val = _format_value(sticky.avg_top_similarity if sticky else None, ".3f")
    if clean and sticky and clean_val and sticky_val:
        return (
            f"Links retrieval precision to carbon. {clean.label} spends only {clean_val} kg per chunk, while {sticky.label} consistently surfaces {sticky_val} similarity contexts, hinting that better ranking trims energy."
        )
    return "Links retrieval carbon-per-chunk against top chunk similarity to flag ranking inefficiencies."


def build_chart_sections(reports: List[ModelReport]) -> List[ChartSection]:
    return [
        ChartSection(
            title="Per-Question Carbon Split",
            description=_describe_carbon_split(reports),
            figure=build_carbon_split_chart(reports),
            div_id="carbon_split",
        ),
        ChartSection(
            title="Accuracy vs Carbon",
            description=_describe_accuracy_vs_carbon(reports),
            figure=build_accuracy_vs_carbon_chart(reports),
            div_id="accuracy_vs_carbon",
        ),
        ChartSection(
            title="Answer Length vs Generation Carbon",
            description=_describe_answer_length_vs_carbon(reports),
            figure=build_answer_length_vs_carbon_chart(reports),
            div_id="answer_length_vs_carbon",
        ),
        ChartSection(
            title="Latency vs Carbon",
            description=_describe_latency_vs_carbon(reports),
            figure=build_latency_vs_carbon_bubble(reports),
            div_id="latency_vs_carbon",
        ),
        ChartSection(
            title="Retrieval Latency vs Carbon",
            description=_describe_retrieval_latency_vs_carbon(reports),
            figure=build_retrieval_latency_vs_carbon(reports),
            div_id="retrieval_vs_carbon",
        ),
        ChartSection(
            title="Retrieval Depth vs Accuracy",
            description=_describe_retrieval_depth_vs_accuracy(reports),
            figure=build_retrieval_depth_vs_accuracy_chart(reports),
            div_id="retrieval_depth_vs_accuracy",
        ),
        ChartSection(
            title="Throughput vs Carbon Intensity",
            description=_describe_throughput_vs_carbon(reports),
            figure=build_throughput_vs_carbon(reports),
            div_id="throughput_vs_carbon",
        ),
        ChartSection(
            title="Carbon per Chunk vs Similarity",
            description=_describe_carbon_per_chunk_vs_similarity(reports),
            figure=build_carbon_per_chunk_vs_similarity_chart(reports),
            div_id="carbon_per_chunk_vs_similarity",
        ),
        ChartSection(
            title="Retrieval-to-Generation Carbon Ratio",
            description=_describe_carbon_ratio(reports),
            figure=build_carbon_ratio_chart(reports),
            div_id="carbon_ratio",
        ),
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def build_insights(reports: List[ModelReport]) -> str:
    bullets: List[str] = []
    best_efficiency = _best_by(reports, key=lambda r: r.generation_carbon_per_100_tokens, reverse=False)
    if best_efficiency:
        bullets.append(
            f"<li><strong>{best_efficiency.label}</strong> has the lowest generation carbon intensity ({best_efficiency.generation_carbon_per_100_tokens:.2e} kg/100 tok).</li>"
        )
    best_accuracy = _best_by(reports, key=lambda r: r.blended_accuracy)
    if best_accuracy:
        bullets.append(
            f"<li><strong>{best_accuracy.label}</strong> leads on blended accuracy ({best_accuracy.blended_accuracy:.3f}).</li>"
        )
    lowest_latency = _best_by(reports, key=lambda r: r.total_latency_ms, reverse=False)
    if lowest_latency:
        bullets.append(
            f"<li><strong>{lowest_latency.label}</strong> is fastest end-to-end ({lowest_latency.total_latency_ms:.1f} ms).</li>"
        )
    best_ratio = _best_by(reports, key=lambda r: r.carbon_ratio, reverse=False)
    if best_ratio:
        bullets.append(
            f"<li><strong>{best_ratio.label}</strong> keeps retrieval carbon proportion lowest (ratio {best_ratio.carbon_ratio:.2f}).</li>"
        )
    leanest_chunk = _best_by(reports, key=lambda r: r.avg_retrieval_carbon_per_chunk, reverse=False)
    if leanest_chunk and leanest_chunk.avg_retrieval_carbon_per_chunk is not None:
        bullets.append(
            f"<li><strong>{leanest_chunk.label}</strong> spends just {leanest_chunk.avg_retrieval_carbon_per_chunk:.2e} kg per retrieved chunk.</li>"
        )
    concise_green = _best_by(reports, key=lambda r: r.generation_carbon_avg, reverse=False)
    if concise_green and concise_green.answer_chars_avg is not None:
        bullets.append(
            f"<li><strong>{concise_green.label}</strong> delivers ~{concise_green.answer_chars_avg:.0f} chars while staying greenest ({concise_green.generation_carbon_avg:.2e} kg/question).</li>"
        )
    if not bullets:
        bullets.append("<li>No insights available (missing metrics).</li>")
    return "<ul>" + "".join(bullets) + "</ul>"


def _best_by(reports: Iterable[ModelReport], key, reverse: bool = True) -> Optional[ModelReport]:
    scored: List[Tuple[float, ModelReport]] = []
    for report in reports:
        value = key(report)
        if isinstance(value, (int, float)):
            scored.append((float(value), report))
    if not scored:
        return None
    value, report = max(scored, key=lambda pair: pair[0]) if reverse else min(scored, key=lambda pair: pair[0])
    return report


def render_dashboard(reports: List[ModelReport], output: Path) -> None:
    if not reports:
        raise SystemExit("No ragres2 summaries found.")

    chart_sections = build_chart_sections(reports)

    card_blocks: List[str] = []
    for section in chart_sections:
        figure_html = pio.to_html(
            section.figure,
            include_plotlyjs=False,
            full_html=False,
            div_id=section.div_id,
            config={"responsive": True},
        )
        card_blocks.append(
            f"<section class=\"card\"><h2>{section.title}</h2><p class=\"description\">{section.description}</p>{figure_html}</section>"
        )

    insights_html = build_insights(reports)
    insights_block = f"<section class=\"card card-wide\"><h2>Quick Insights</h2>{insights_html}</section>"

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>RAG Benchmark Dashboard (ragres2)</title>
<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 0; padding: 2rem; background: #f6f8fb; }}
h1 {{ text-align: center; margin-bottom: 0.25rem; }}
.subtitle {{ text-align: center; color: #4b5563; margin-bottom: 2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }}
.card {{ background: #fff; border-radius: 0.75rem; padding: 1.25rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); }}
.card.card-wide {{ grid-column: 1 / -1; }}
.description {{ margin-top: 0; color: #475569; }}
.card .plotly-graph-div {{ width: 100% !important; }}
ul {{ margin: 0; padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>RAG Benchmark Overview (ragres2)</h1>
<p class=\"subtitle\">Carbon-aware retrieval + generation diagnostics for ragres2 runs.</p>
<div class=\"grid\">
{insights_block}
{"".join(card_blocks)}
</div>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.rag_root).resolve()
    reports = collect_reports(root)
    render_dashboard(reports, Path(args.output).resolve())


if __name__ == "__main__":
    main()
