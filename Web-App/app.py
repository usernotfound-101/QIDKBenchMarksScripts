import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from data_loader import load_multiple_metrics
from evaluation_loader import load_evaluation_file

st.set_page_config(page_title="SLM Metrics Dashboard", layout="wide")

st.title("SLM Model-Task Evaluation Dashboard")

with st.sidebar:
    st.subheader("Mode")
    mode = st.selectbox("Select Mode", ["Analysis", "Comparison"], index=0, help="Analysis: single model/task deep dive. Comparison: compare multiple models.")

# Optional auto-refresh
# Auto refresh removed per request

with st.sidebar:
    st.header("Model & Task Selection")
    model_options = ["phi", "gemma", "qwen", "llama"]
    task_options = ["Q/A", "summarisation"]
    if mode == 'Analysis':
        default_model = st.session_state.get('current_model', 'phi')
        if default_model not in model_options:
            default_model = 'phi'
        model_name = st.selectbox("Model", options=model_options, index=model_options.index(default_model))
    else:
        st.markdown("Select models to compare:")
        selected_models = []
        cmp_uploads = {}
        for m in model_options:
            checked = st.checkbox(m, key=f'cmp_cb_{m}')
            if checked:
                selected_models.append(m)
                csv_up = st.file_uploader(f"{m} Resource CSV", type=['csv'], accept_multiple_files=False, key=f'cmp_csv_{m}')
                json_up = st.file_uploader(f"{m} Evaluation JSON/JSONL", type=['json','jsonl'], accept_multiple_files=False, key=f'cmp_json_{m}')
                cmp_uploads[m] = {'csv': csv_up, 'json': json_up}
        st.session_state['cmp_models'] = selected_models
        st.session_state['cmp_uploads'] = cmp_uploads
        model_name = None

    default_task = st.session_state.get('current_task_display', 'summarisation')
    if default_task not in task_options:
        default_task = 'summarisation'
    task_display = st.selectbox("Task", options=task_options, index=task_options.index(default_task))

    st.session_state['current_model'] = model_name
    st.session_state['current_task_display'] = task_display
    # Track selection changes and use dynamic keys so uploaders reset when selection changes
    prev_pair = st.session_state.get('last_pair')
    current_pair = (model_name, task_display)
    selection_changed = prev_pair is not None and prev_pair != current_pair
    st.session_state['last_pair'] = current_pair
    st.markdown("---")
    st.subheader("Upload Resource Metrics CSV")
    if mode == 'Analysis':
        key_suffix = f"{model_name}_{task_display}"
        res_csv = st.file_uploader("Resource CSV", type=['csv'], accept_multiple_files=False, key=f'res_csv_{key_suffix}')
    else:
        res_csv = None  # Per-model uploaders are shown above
    st.caption("CSV may contain temperature, memory_used_mb, cpu_percent columns plus timestamp, model, task. Accuracy now optional.")
    st.markdown("---")
    st.subheader("Upload Evaluation JSON/JSONL")
    if mode == 'Analysis':
        # For Q/A, we ignore evaluation uploads and use hardcoded metrics; summarisation unchanged
        if task_display == 'Q/A':
            eval_file = None
        else:
            eval_file = st.file_uploader("Evaluation File", type=['json','jsonl'], accept_multiple_files=False, key=f'eval_single_{key_suffix}', help="Single file for this model-task")
    else:
        eval_file = None  # Per-model uploaders are shown above
    eval_data = None
    if eval_file:
        tmp_eval = Path(f"/tmp/eval_current_{eval_file.name}")
        tmp_eval.write_bytes(eval_file.getbuffer())
        try:
            eval_data = load_evaluation_file(tmp_eval)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
    # Hardcoded Q/A metrics per model
    qa_metrics_map = {
        'gemma': {'rougeL_f1': 0.3805, 'bertscore_f1': 0.8721},
        'phi':   {'rougeL_f1': 0.4415, 'bertscore_f1': 0.8839},
        'qwen':  {'rougeL_f1': 0.3477, 'bertscore_f1': 0.8641},
        'llama': {'rougeL_f1': 0.3060, 'bertscore_f1': 0.8594},
    }

# Load dataframe
@st.cache_data(show_spinner=True)
def load_resource_df(uploaded, model, task_display):
    if not uploaded:
        return pd.DataFrame(columns=["timestamp","model","task","temperature","memory_used_mb","cpu_percent"])
    tmp = Path(f"/tmp/res_metrics_{uploaded.name}")
    tmp.write_bytes(uploaded.getbuffer())
    try:
        raw = pd.read_csv(tmp)
        # Detect generic schema (Timestamp, Category, Metric, Value) used by phi-summ.csv
        # Normalize column names for robustness
        norm_cols = {c.lower(): c for c in raw.columns}
        has_generic = all(k in norm_cols for k in ["timestamp","category","metric","value"])
        if has_generic and not {"timestamp","model","task"}.issubset(raw.columns):
            # Keep raw for advanced processing; return both derived and raw in a dict-like DataFrame container
            # Build a normalized view
            ts_col = norm_cols['timestamp']
            cat_col = norm_cols['category']
            met_col = norm_cols['metric']
            val_col = norm_cols['value']
            raw = raw.rename(columns={ts_col:'Timestamp', cat_col:'Category', met_col:'Metric', val_col:'Value'})
            # Normalize text values (lowercase, strip)
            raw['Category'] = raw['Category'].astype(str).str.strip().str.lower()
            raw['Metric'] = raw['Metric'].astype(str).str.strip().str.lower()
            raw = raw.sort_values('Timestamp')
            raw['__model'] = model
            raw['__task'] = 'qa' if task_display == 'Q/A' else 'summarization'
            # Derive aggregated table for existing charts
            # Accept category/metric variants
            cpu_rows = raw[raw['Category'].isin([
                'cpu core utilization','cpu','cpu_utilization','cpu_core_utilization','cpu core','cpu_core','core utilization'
            ])]
            cpu_pivot = cpu_rows.pivot(index='Timestamp', columns='Metric', values='Value') if not cpu_rows.empty else pd.DataFrame()
            cpu_mean = cpu_pivot.mean(axis=1) if not cpu_pivot.empty else pd.Series(dtype=float)
            mem_rows = raw[(raw['Category'].isin(['system memory','memory'])) & (raw['Metric'].isin(['used memory','used_mem','used']))]
            mem_series = mem_rows.set_index('Timestamp')['Value'] if not mem_rows.empty else pd.Series(dtype=float)
            therm_rows = raw[raw['Category'].isin(['thermal','temperature','thermals'])]
            temp_series = therm_rows.groupby('Timestamp')['Value'].mean() if not therm_rows.empty else pd.Series(dtype=float)
            all_ts = sorted(set(cpu_mean.index) | set(mem_series.index) | set(temp_series.index))
            data = pd.DataFrame({'timestamp': all_ts})
            if not cpu_mean.empty:
                data = data.merge(cpu_mean.rename('cpu_percent'), left_on='timestamp', right_index=True, how='left')
            if not mem_series.empty:
                if mem_series.median() > 1e6:
                    mem_conv = mem_series / (1024**2)
                else:
                    mem_conv = mem_series
                data = data.merge(mem_conv.rename('memory_used_mb'), left_on='timestamp', right_index=True, how='left')
            if not temp_series.empty:
                data = data.merge(temp_series.rename('temperature'), left_on='timestamp', right_index=True, how='left')
            data['model'] = model
            data['task'] = raw['__task'].iloc[0] if not raw.empty else ('qa' if task_display == 'Q/A' else 'summarization')
            if pd.api.types.is_numeric_dtype(data['timestamp']):
                base = pd.Timestamp.utcnow().floor('s')
                rel = data['timestamp'] - data['timestamp'].min()
                if rel.max() < 1e11:
                    data['timestamp'] = base + pd.to_timedelta(rel, unit='us')
                else:
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us', origin='unix')
            # Attach raw CPU pivot for per-core plotting via attribute
            # Store auxiliary structures in session state keyed by upload name to avoid serialization issues
            aux_key = f"aux_{uploaded.name}_{model}_{task_display}".replace('-','_')
            st.session_state[aux_key] = {
                'cpu_pivot': cpu_pivot,
                'raw_generic': raw
            }
            data['__aux_key'] = aux_key
            return data.sort_values('timestamp')
        else:
            # Use existing loader path
            df = load_multiple_metrics([tmp])
            if 'model' in df.columns:
                df = df[df['model'] == model]
            if 'task' in df.columns:
                task_l = df['task'].astype(str).str.lower()
                if task_display == 'Q/A':
                    accepted = {"q/a", "qa", "question answering", "question_answering"}
                else:
                    accepted = {"summarisation", "summarization", "summary"}
                df = df[task_l.isin(accepted)]
            return df.sort_values('timestamp')
    except Exception as e:
        st.error(f"Failed to load resource metrics: {e}")
        return pd.DataFrame(columns=["timestamp","model","task","temperature","memory_used_mb","cpu_percent"])

resources_df = load_resource_df(res_csv, model_name, task_display) if mode == 'Analysis' else pd.DataFrame()

if mode == 'Comparison':
    pass
elif eval_data is None or res_csv is None:
    st.warning("In Analysis mode you must upload both a resource CSV and an evaluation JSON file.")

if mode == 'Analysis' and eval_data and not resources_df.empty:
    header_cols = st.columns(4)
    if eval_data['type'] == 'qa':
        em = eval_data['metrics'].get('exact_match', {}).get('mean', float('nan'))
        f1m = eval_data['metrics'].get('f1', {}).get('mean', float('nan'))
        bert_f1 = eval_data['metrics'].get('bertscore_f1', {}).get('mean', float('nan'))
        header_cols[0].metric("EM Mean", f"{em:.3f}" if pd.notna(em) else "N/A")
        header_cols[1].metric("F1 Mean", f"{f1m:.3f}" if pd.notna(f1m) else "N/A")
        header_cols[2].metric("BERT F1 Mean", f"{bert_f1:.3f}" if pd.notna(bert_f1) else "N/A")
        header_cols[3].metric("Count", eval_data['metrics'].get('count', 0))
    elif eval_data['type'] == 'summarization':
        rougeL = eval_data['metrics'].get('rougeL_f1', {}).get('mean', float('nan'))
        bert_f1 = eval_data['metrics'].get('bertscore_f1', {}).get('mean', float('nan'))
        cosine = eval_data['metrics'].get('cosine_similarity', {}).get('mean', float('nan'))
        header_cols[0].metric("ROUGE-L F1", f"{rougeL:.3f}" if pd.notna(rougeL) else "N/A")
        header_cols[1].metric("BERT F1", f"{bert_f1:.3f}" if pd.notna(bert_f1) else "N/A")
        header_cols[2].metric("Cosine", f"{cosine:.3f}" if pd.notna(cosine) else "N/A")
        header_cols[3].metric("Count", eval_data['metrics'].get('count', 0))
    else:
        header_cols[0].metric("Type", eval_data['type'])
        header_cols[1].metric("Count", eval_data['metrics'].get('count', 0))
    header_cols[2].metric("Model", model_name)
    header_cols[3].metric("Task", task_display)

if mode == 'Analysis':
    metrics_tabs = st.tabs(["Resources Temperature", "Resources Memory", "Raw Resources", "Evaluation JSON", "Evaluation Plots", "CPU Cores", "CO2 Emissions"])
else:
    metrics_tabs = []

if mode == 'Analysis':
    with metrics_tabs[0]:
        plotted = False
        if not resources_df.empty and 'temperature' in resources_df.columns:
            temp_col = resources_df['temperature']
            if temp_col.notna().any():
                fig_temp = px.line(resources_df.dropna(subset=['temperature']), x='timestamp', y='temperature', title='Temperature vs Time')
                st.plotly_chart(fig_temp, use_container_width=True)
                plotted = True
        if not plotted:
            # Fallback: derive from raw generic
            if not resources_df.empty and '__aux_key' in resources_df.columns:
                aux = st.session_state.get(resources_df['__aux_key'].iloc[0])
                if aux and aux.get('raw_generic') is not None:
                    rawg = aux['raw_generic']
                    therm = rawg[rawg['Category'] == 'Thermal']
                    if not therm.empty:
                        temp_series = therm.groupby('Timestamp')['Value'].mean()
                        tmp_df = temp_series.reset_index().rename(columns={'Timestamp':'timestamp','Value':'temperature'})
                        # Convert timestamp to datetime if numeric
                        if pd.api.types.is_numeric_dtype(tmp_df['timestamp']):
                            base = pd.Timestamp.utcnow().floor('s')
                            rel = tmp_df['timestamp'] - tmp_df['timestamp'].min()
                            tmp_df['timestamp'] = base + pd.to_timedelta(rel, unit='us')
                        fig_temp = px.line(tmp_df, x='timestamp', y='temperature', title='Temperature vs Time')
                        st.plotly_chart(fig_temp, use_container_width=True)
                        plotted = True
        if not plotted:
            st.info("No temperature data available to plot.")
        # Temperature statistics
        if not resources_df.empty and 'temperature' in resources_df.columns and resources_df['temperature'].notna().any():
            s = resources_df['temperature'].dropna()
            stat_cols = st.columns(5)
            stat_cols[0].metric('Avg', f"{s.mean():.3f}")
            stat_cols[1].metric('Median', f"{s.median():.3f}")
            stat_cols[2].metric('Min', f"{s.min():.3f}")
            stat_cols[3].metric('Max', f"{s.max():.3f}")
            stat_cols[4].metric('Range', f"{(s.max()-s.min()):.3f}")
    with metrics_tabs[1]:
        if resources_df.empty or 'memory_used_mb' not in resources_df.columns:
            st.info("No memory data.")
        else:
            fig_mem = px.line(resources_df, x='timestamp', y='memory_used_mb', title='Memory Usage (MB) Over Time')
            st.plotly_chart(fig_mem, use_container_width=True)
    with metrics_tabs[2]:
        if resources_df.empty:
            st.dataframe(pd.DataFrame(columns=['timestamp','model','task','temperature','memory_used_mb','cpu_percent']))
        else:
            cols = [c for c in ['timestamp','model','task','temperature','memory_used_mb','cpu_percent'] if c in resources_df.columns]
            st.dataframe(resources_df[cols])
    with metrics_tabs[3]:
        if not eval_data:
            st.info("No evaluation uploaded.")
        else:
            st.json(eval_data)
    with metrics_tabs[4]:
        if not eval_data or not eval_data.get('metrics'):
            st.info("No evaluation to plot.")
        else:
            m = eval_data['metrics']
            per = m.get('per_item', {})
            # Summary stats section
            def series_mean(name_dict_key: str, metric_key: str):
                # Prefer aggregate mean from metrics, else compute from per-item
                agg = m.get(metric_key, {}).get('mean')
                if pd.notna(agg):
                    return float(agg)
                arr = per.get(name_dict_key)
                return float(pd.Series(arr).mean()) if arr else float('nan')
            def series_stats(arr):
                s = pd.Series(arr).dropna() if arr else pd.Series(dtype=float)
                if s.empty:
                    return { 'mean': float('nan'), 'mode': float('nan'), 'range': float('nan'), 'min': float('nan'), 'max': float('nan') }
                mode_vals = s.round(6).mode()
                mode_val = float(mode_vals.iloc[0]) if not mode_vals.empty else float('nan')
                return {
                    'mean': float(s.mean()),
                    'mode': mode_val,
                    'range': float(s.max() - s.min()),
                    'min': float(s.min()),
                    'max': float(s.max()),
                }
            # Compute stats for available series
            # Use hardcoded Q/A metrics if task is Q/A
            if task_display == 'Q/A':
                hard = qa_metrics_map.get(model_name or '', {})
                rouge_items = [hard.get('rougeL_f1')] if hard else None
                bert_items = [hard.get('bertscore_f1')] if hard else None
                cos_items = None
            else:
                rouge_items = per.get('rougeL_f1')
                bert_items = per.get('bertscore_f1')
                cos_items = per.get('cosine_similarity')
            rouge_stats = series_stats(rouge_items)
            bert_stats = series_stats(bert_items)
            cos_stats = series_stats(cos_items)
            st.subheader("Evaluation Summary")
            cols = st.columns(3)
            # ROUGE-L F1
            with cols[0]:
                st.markdown("**ROUGE-L F1**")
                st.metric("Mean", f"{rouge_stats['mean']:.3f}" if pd.notna(rouge_stats['mean']) else "N/A")
                st.metric("Mode", f"{rouge_stats['mode']:.3f}" if pd.notna(rouge_stats['mode']) else "N/A")
                st.metric("Range", f"{rouge_stats['range']:.3f}" if pd.notna(rouge_stats['range']) else "N/A")
                st.metric("Min/Max", f"{rouge_stats['min']:.3f} / {rouge_stats['max']:.3f}" if pd.notna(rouge_stats['min']) and pd.notna(rouge_stats['max']) else "N/A")
            # BERTScore F1
            with cols[1]:
                st.markdown("**BERTScore F1**")
                st.metric("Mean", f"{bert_stats['mean']:.3f}" if pd.notna(bert_stats['mean']) else "N/A")
                st.metric("Mode", f"{bert_stats['mode']:.3f}" if pd.notna(bert_stats['mode']) else "N/A")
                st.metric("Range", f"{bert_stats['range']:.3f}" if pd.notna(bert_stats['range']) else "N/A")
                st.metric("Min/Max", f"{bert_stats['min']:.3f} / {bert_stats['max']:.3f}" if pd.notna(bert_stats['min']) and pd.notna(bert_stats['max']) else "N/A")
            # Cosine
            with cols[2]:
                st.markdown("**Cosine Similarity**")
                st.metric("Mean", f"{cos_stats['mean']:.3f}" if pd.notna(cos_stats['mean']) else "N/A")
                st.metric("Mode", f"{cos_stats['mode']:.3f}" if pd.notna(cos_stats['mode']) else "N/A")
                st.metric("Range", f"{cos_stats['range']:.3f}" if pd.notna(cos_stats['range']) else "N/A")
                st.metric("Min/Max", f"{cos_stats['min']:.3f} / {cos_stats['max']:.3f}" if pd.notna(cos_stats['min']) and pd.notna(cos_stats['max']) else "N/A")
            if not per:
                st.info("No per-item scores available. Install optional deps for BERTScore/cosine to enable plots.")
            else:
                if task_display == 'Q/A':
                    if bert_items:
                        df_b = pd.DataFrame({'index': [1], 'bertscore_f1': bert_items})
                        fig_b = px.bar(df_b, x='index', y='bertscore_f1', title='BERTScore F1 (Q/A hardcoded)')
                        st.plotly_chart(fig_b, use_container_width=True)
                    if rouge_items:
                        df_r = pd.DataFrame({'index': [1], 'rougeL_f1': rouge_items})
                        fig_r = px.bar(df_r, x='index', y='rougeL_f1', title='ROUGE-L F1 (Q/A hardcoded)')
                        st.plotly_chart(fig_r, use_container_width=True)
                else:
                    if 'bertscore_f1' in per:
                        df_b = pd.DataFrame({
                        'index': list(range(1, len(per['bertscore_f1']) + 1)),
                        'bertscore_f1': per['bertscore_f1']
                    })
                    fig_b = px.line(df_b, x='index', y='bertscore_f1', title='Per-item BERTScore F1')
                    st.plotly_chart(fig_b, use_container_width=True)
                if task_display != 'Q/A' and 'cosine_similarity' in per:
                    df_c = pd.DataFrame({
                        'index': list(range(1, len(per['cosine_similarity']) + 1)),
                        'cosine_similarity': per['cosine_similarity']
                    })
                    fig_c = px.line(df_c, x='index', y='cosine_similarity', title='Per-item Cosine Similarity')
                    st.plotly_chart(fig_c, use_container_width=True)
                # ROUGE-L F1 per-item plot for summarization
                if task_display != 'Q/A' and 'rougeL_f1' in per:
                    df_r = pd.DataFrame({
                        'index': list(range(1, len(per['rougeL_f1']) + 1)),
                        'rougeL_f1': per['rougeL_f1']
                    })
                    fig_r = px.line(df_r, x='index', y='rougeL_f1', title='Per-item ROUGE-L F1')
                    st.plotly_chart(fig_r, use_container_width=True)
    # CPU Cores tab
    with metrics_tabs[5]:
        cpu_pivot = None
        if not resources_df.empty and '__aux_key' in resources_df.columns:
            aux = st.session_state.get(resources_df['__aux_key'].iloc[0])
            if aux:
                cpu_pivot = aux.get('cpu_pivot')
        if cpu_pivot is None or cpu_pivot.empty:
            st.info("No per-core CPU data available.")
        else:
            st.subheader("Per-Core CPU Utilization")
            for col in cpu_pivot.columns:
                core_df = cpu_pivot[[col]].reset_index().rename(columns={'Timestamp':'timestamp', col:'util'})
                fig_core = px.line(core_df, x='timestamp', y='util', title=f"{col} vs Time")
                st.plotly_chart(fig_core, use_container_width=True)
            # Stats table
            stats_rows = []
            for col in cpu_pivot.columns:
                s = cpu_pivot[col].dropna()
                if s.empty:
                    continue
                stats_rows.append({
                    'core': col,
                    'mean': s.mean(),
                    'median': s.median(),
                    'min': s.min(),
                    'max': s.max(),
                    'range': s.max()-s.min()
                })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows))
    # CO2 Emissions tab
    with metrics_tabs[6]:
        raw_generic = None
        if not resources_df.empty and '__aux_key' in resources_df.columns:
            aux = st.session_state.get(resources_df['__aux_key'].iloc[0])
            if aux:
                raw_generic = aux.get('raw_generic')
        if raw_generic is None or raw_generic.empty:
            # Fallback: try to compute CO2 directly from resources_df (aggregated series)
            if not resources_df.empty and all(c in resources_df.columns for c in ['timestamp','cpu_percent','memory_used_mb','temperature']):
                co2_df = resources_df[['timestamp','cpu_percent','memory_used_mb','temperature']].copy()
                co2_df = co2_df.dropna(subset=['timestamp']).sort_values('timestamp')
                # Time axis in seconds since start
                if pd.api.types.is_datetime64_any_dtype(co2_df['timestamp']):
                    t0 = co2_df['timestamp'].min()
                    co2_df['time_sec'] = (co2_df['timestamp'] - t0).dt.total_seconds()
                else:
                    ts = pd.to_numeric(co2_df['timestamp'], errors='coerce')
                    co2_df['time_sec'] = (ts - ts.min()) / (1e6 if (ts.max() - ts.min()) > 1e6 else 1.0)
                # Power model
                P_IDLE_W = 0.5; P_MAX_W = 13.0; T_AMBIENT_C = 25.0
                K_THERM_W_PER_C = 0.2; RAM_W_PER_GB = 0.375; CARBON_INTENSITY_KG_PER_KWH = 0.708
                co2_df['U'] = pd.to_numeric(co2_df['cpu_percent'], errors='coerce') / 100.0
                co2_df['P_cpu_W'] = P_IDLE_W + (P_MAX_W - P_IDLE_W) * co2_df['U'].fillna(0.0)
                co2_df['Used_Mem_GB'] = pd.to_numeric(co2_df['memory_used_mb'], errors='coerce').fillna(0.0) / 1024.0
                co2_df['P_ram_W'] = co2_df['Used_Mem_GB'] * RAM_W_PER_GB
                delta_T = (pd.to_numeric(co2_df['temperature'], errors='coerce').fillna(T_AMBIENT_C) - T_AMBIENT_C).clip(lower=0)
                co2_df['P_thermal_W'] = K_THERM_W_PER_C * delta_T
                co2_df['P_total_W'] = co2_df['P_cpu_W'] + co2_df['P_ram_W'] + co2_df['P_thermal_W']
                co2_df = co2_df.sort_values('time_sec')
                co2_df['delta_t_sec'] = co2_df['time_sec'].diff().fillna(0.0)
                co2_df['energy_kWh'] = (co2_df['P_total_W'] / 1000.0) * (co2_df['delta_t_sec'] / 3600.0)
                co2_df['cum_energy_kWh'] = co2_df['energy_kWh'].cumsum()
                co2_df['CO2_kg'] = co2_df['energy_kWh'] * CARBON_INTENSITY_KG_PER_KWH
                co2_df['cum_CO2_kg'] = co2_df['CO2_kg'].cumsum()
                co2_df['CO2_rate_kg_per_s'] = co2_df['CO2_kg'] / co2_df['delta_t_sec'].replace(0, pd.NA)
                co2_df['CO2_rate_g_per_s'] = co2_df['CO2_rate_kg_per_s'] * 1000.0
                # Plots
                fig_co2 = px.line(co2_df, x='time_sec', y=co2_df['cum_CO2_kg']*1000.0, title='Cumulative CO2 Emissions (g)')
                st.plotly_chart(fig_co2, use_container_width=True)
                fig_rate = px.line(co2_df.dropna(subset=['CO2_rate_g_per_s']), x='time_sec', y='CO2_rate_g_per_s', title='CO2 Emission Rate (g/s)')
                st.plotly_chart(fig_rate, use_container_width=True)
                total_co2_g = co2_df['cum_CO2_kg'].iloc[-1] * 1000.0
                total_energy = co2_df['cum_energy_kWh'].iloc[-1]
                stat_cols = st.columns(6)
                stat_cols[0].metric('Total CO2 (g)', f"{total_co2_g:.3f}")
                stat_cols[1].metric('Total Energy (kWh)', f"{total_energy:.6f}")
                rate_series = co2_df['CO2_rate_g_per_s'].dropna()
                if not rate_series.empty:
                    stat_cols[2].metric('Rate Mean (g/s)', f"{rate_series.mean():.6f}")
                    stat_cols[3].metric('Median', f"{rate_series.median():.6f}")
                    stat_cols[4].metric('Min/Max', f"{rate_series.min():.6f} / {rate_series.max():.6f}")
                    stat_cols[5].metric('Range', f"{(rate_series.max()-rate_series.min()):.6f}")
            else:
                st.info("No raw generic data for CO2 computation.")
        else:
            # Reproduce co2.py logic
            P_IDLE_W = 0.5
            P_MAX_W = 13.0
            T_AMBIENT_C = 25.0
            K_THERM_W_PER_C = 0.2
            RAM_W_PER_GB = 0.375
            CARBON_INTENSITY_KG_PER_KWH = 0.708
            rg = raw_generic.copy()
            # Ensure lowercase to match normalization
            if 'Category' in rg.columns and rg['Category'].dtype == object:
                rg['Category'] = rg['Category'].astype(str).str.strip().str.lower()
            if 'Metric' in rg.columns and rg['Metric'].dtype == object:
                rg['Metric'] = rg['Metric'].astype(str).str.strip().str.lower()

            cpu_rows = rg[rg['Category'].isin([
                'cpu core utilization','cpu','cpu_utilization','cpu_core_utilization','cpu core','cpu_core','core utilization'
            ])]
            cpu_group = cpu_rows.groupby('Timestamp')['Value'].mean().rename('CPU_Util_pct')
            mem_rows = rg[(rg['Category'].isin(['system memory','memory'])) & (rg['Metric'].isin(['used memory','used_mem','used']))]
            mem_group = mem_rows.groupby('Timestamp')['Value'].mean().rename('Used_Mem_bytes')
            temp_rows = rg[rg['Category'].isin(['thermal','temperature','thermals','temp'])]
            temp_group = temp_rows.groupby('Timestamp')['Value'].mean().rename('Temp_C')

            # Build a common time index from any available series
            series_list = [s for s in [cpu_group, mem_group, temp_group] if s is not None and not s.empty]
            if len(series_list) == 0:
                st.info("Insufficient data for CO2 model.")
                series_list = []  # continue without plotting
            common_idx = pd.Index(sorted(set().union(*[set(s.index) for s in series_list])))

            # Fallbacks: if a series is missing, create reasonable defaults aligned to common_idx
            if cpu_group is None or cpu_group.empty:
                # Assume minimal idle utilization (0%)
                cpu_group = pd.Series(0.0, index=common_idx, name='CPU_Util_pct')
            else:
                cpu_group = cpu_group.reindex(common_idx).ffill().bfill()

            if mem_group is None or mem_group.empty:
                # If memory not available, assume 0 additional RAM power
                mem_group = pd.Series(0.0, index=common_idx, name='Used_Mem_bytes')
            else:
                mem_group = mem_group.reindex(common_idx).ffill().bfill()

            if temp_group is None or temp_group.empty:
                # If temperature not available, assume ambient temperature
                temp_group = pd.Series(T_AMBIENT_C, index=common_idx, name='Temp_C')
            else:
                temp_group = temp_group.reindex(common_idx).ffill().bfill()

            co2_df = pd.concat([cpu_group, mem_group, temp_group], axis=1)
            if co2_df.empty or len(series_list) == 0:
                st.info("Insufficient data for CO2 model.")
            else:
                # Robust time axis in seconds since start
                idx = co2_df.index
                if pd.api.types.is_integer_dtype(idx) or pd.api.types.is_float_dtype(idx):
                    idx = pd.to_numeric(idx)
                    scale = 1.0
                    # Heuristic: values look like microseconds if diffs are large
                    if (idx.max() - idx.min()) > 1e6:
                        scale = 1e6
                    co2_df['time_sec'] = (idx - idx.min()) / scale
                else:
                    idx_dt = pd.to_datetime(idx, errors='coerce')
                    delta = (idx_dt - idx_dt.min())
                    co2_df['time_sec'] = delta.dt.total_seconds()
                co2_df['U'] = co2_df['CPU_Util_pct'] / 100.0
                co2_df['P_cpu_W'] = P_IDLE_W + (P_MAX_W - P_IDLE_W) * co2_df['U']
                co2_df['Used_Mem_GB'] = co2_df['Used_Mem_bytes'] / (1024.0 ** 3)
                co2_df['P_ram_W'] = co2_df['Used_Mem_GB'] * RAM_W_PER_GB
                delta_T = (co2_df['Temp_C'] - T_AMBIENT_C).clip(lower=0)
                co2_df['P_thermal_W'] = K_THERM_W_PER_C * delta_T
                co2_df['P_total_W'] = co2_df['P_cpu_W'] + co2_df['P_ram_W'] + co2_df['P_thermal_W']
                co2_df['time_sec'] = co2_df['time_sec'].astype(float)
                co2_df = co2_df.sort_values('time_sec')
                co2_df['delta_t_sec'] = co2_df['time_sec'].diff().fillna(0.0)
                co2_df['energy_kWh'] = (co2_df['P_total_W'] / 1000.0) * (co2_df['delta_t_sec'] / 3600.0)
                co2_df['cum_energy_kWh'] = co2_df['energy_kWh'].cumsum()
                co2_df['CO2_kg'] = co2_df['energy_kWh'] * CARBON_INTENSITY_KG_PER_KWH
                co2_df['cum_CO2_kg'] = co2_df['CO2_kg'].cumsum()
                co2_df['CO2_rate_kg_per_s'] = co2_df['CO2_kg'] / co2_df['delta_t_sec'].replace(0, pd.NA)
                co2_df['CO2_rate_g_per_s'] = co2_df['CO2_rate_kg_per_s'] * 1000.0
                # Precompute y columns to avoid confusion and ensure correct mapping
                co2_df['cum_CO2_g'] = co2_df['cum_CO2_kg'] * 1000.0
                fig_co2 = px.line(co2_df, x='time_sec', y='cum_CO2_g', title='Cumulative CO2 Emissions (g)')
                st.plotly_chart(fig_co2, use_container_width=True)
                fig_rate = px.line(co2_df.dropna(subset=['CO2_rate_g_per_s']), x='time_sec', y='CO2_rate_g_per_s', title='CO2 Emission Rate (g/s)')
                st.plotly_chart(fig_rate, use_container_width=True)
                total_co2_g = co2_df['cum_CO2_kg'].iloc[-1] * 1000.0
                total_energy = co2_df['cum_energy_kWh'].iloc[-1]
                stat_cols = st.columns(6)
                stat_cols[0].metric('Total CO2 (g)', f"{total_co2_g:.3f}")
                stat_cols[1].metric('Total Energy (kWh)', f"{total_energy:.6f}")
                rate_series = co2_df['CO2_rate_g_per_s'].dropna()
                if not rate_series.empty:
                    stat_cols[2].metric('Mean Rate (g/s)', f"{rate_series.mean():.6f}")
                    stat_cols[3].metric('Median Rate', f"{rate_series.median():.6f}")
                    stat_cols[4].metric('Min/Max', f"{rate_series.min():.6f} / {rate_series.max():.6f}")
                    mode_vals = rate_series.round(6).mode()
                    mode_val = mode_vals.iloc[0] if not mode_vals.empty else float('nan')
                    stat_cols[5].metric('Mode Rate', f"{mode_val:.6f}")
                st.dataframe(co2_df[['time_sec','P_total_W','energy_kWh','cum_energy_kWh','CO2_kg','cum_CO2_kg']].tail(20))

# Allow download of resources subset if provided
if mode == 'Analysis' and not resources_df.empty:
    csv_bytes = resources_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Resources CSV", data=csv_bytes, file_name='resources_subset.csv', mime='text/csv')

if mode == 'Analysis':
    st.caption("Analysis mode: upload both resource CSV and evaluation JSON for selected model/task.")
else:
    # Comparison mode implementation: compute metrics per model and rank
    selected_models = st.session_state.get('cmp_models', [])
    cmp_uploads = st.session_state.get('cmp_uploads', {})
    if not selected_models:
        st.warning("Select at least 2 models and upload CSV/JSON for each.")
    else:
        missing = [m for m in selected_models if not cmp_uploads.get(m, {}).get('csv') or not cmp_uploads.get(m, {}).get('json')]
        if missing:
            st.warning(f"Provide one CSV and one JSON for: {', '.join(missing)}")
        else:
            st.subheader("Comparison Results")
            rows = []
            for model in selected_models:
                rfile = cmp_uploads[model]['csv']
                efile = cmp_uploads[model]['json']
                # Load resource and eval for this model
                rdf = load_resource_df(rfile, model, task_display)
                try:
                    etmp = Path(f"/tmp/cmp_eval_{model}_{efile.name}")
                    etmp.write_bytes(efile.getbuffer())
                    edata = load_evaluation_file(etmp)
                except Exception as e:
                    st.error(f"Evaluation failed for {model}: {e}")
                    continue
                # Compute average CO2 rate (g/s)
                avg_co2 = float('nan')
                rawg = None
                if not rdf.empty and '__aux_key' in rdf.columns:
                    aux = st.session_state.get(rdf['__aux_key'].iloc[0])
                    if aux:
                        rawg = aux.get('raw_generic')
                def compute_avg_co2_from_raw(raw_generic_df):
                    if raw_generic_df is None or raw_generic_df.empty:
                        return float('nan')
                    rg = raw_generic_df.copy()
                    rg['Category'] = rg['Category'].astype(str).str.strip().str.lower()
                    rg['Metric'] = rg['Metric'].astype(str).str.strip().str.lower()
                    cpu_rows = rg[rg['Category'].isin(['cpu core utilization','cpu','cpu_utilization','cpu_core_utilization','cpu core','cpu_core','core utilization'])]
                    cpu_group = cpu_rows.groupby('Timestamp')['Value'].mean().rename('CPU_Util_pct')
                    mem_rows = rg[(rg['Category'].isin(['system memory','memory'])) & (rg['Metric'].isin(['used memory','used_mem','used']))]
                    mem_group = mem_rows.groupby('Timestamp')['Value'].mean().rename('Used_Mem_bytes')
                    temp_rows = rg[rg['Category'].isin(['thermal','temperature','thermals','temp'])]
                    temp_group = temp_rows.groupby('Timestamp')['Value'].mean().rename('Temp_C')
                    series_list = [s for s in [cpu_group, mem_group, temp_group] if s is not None and not s.empty]
                    if not series_list:
                        return float('nan')
                    common_idx = pd.Index(sorted(set().union(*[set(s.index) for s in series_list])))
                    # Defaults
                    if cpu_group is None or cpu_group.empty:
                        cpu_group = pd.Series(0.0, index=common_idx, name='CPU_Util_pct')
                    else:
                        cpu_group = cpu_group.reindex(common_idx).ffill().bfill()
                    if mem_group is None or mem_group.empty:
                        mem_group = pd.Series(0.0, index=common_idx, name='Used_Mem_bytes')
                    else:
                        mem_group = mem_group.reindex(common_idx).ffill().bfill()
                    if temp_group is None or temp_group.empty:
                        T_AMBIENT_C = 25.0
                        temp_group = pd.Series(T_AMBIENT_C, index=common_idx, name='Temp_C')
                    else:
                        temp_group = temp_group.reindex(common_idx).ffill().bfill()
                    co2_df = pd.concat([cpu_group, mem_group, temp_group], axis=1)
                    # time axis
                    idx = co2_df.index
                    if pd.api.types.is_integer_dtype(idx) or pd.api.types.is_float_dtype(idx):
                        idx = pd.to_numeric(idx)
                        scale = 1.0
                        if (idx.max() - idx.min()) > 1e6:
                            scale = 1e6
                        co2_df['time_sec'] = (idx - idx.min()) / scale
                    else:
                        idx_dt = pd.to_datetime(idx, errors='coerce')
                        delta = (idx_dt - idx_dt.min())
                        co2_df['time_sec'] = delta.dt.total_seconds()
                    # power and CO2
                    P_IDLE_W, P_MAX_W = 1.2, 8.0
                    T_AMBIENT_C, K_THERM_W_PER_C = 25.0, 0.2
                    RAM_W_PER_GB, CI = 0.375, 0.708
                    co2_df['U'] = co2_df['CPU_Util_pct'] / 100.0
                    co2_df['P_cpu_W'] = P_IDLE_W + (P_MAX_W - P_IDLE_W) * co2_df['U']
                    co2_df['Used_Mem_GB'] = co2_df['Used_Mem_bytes'] / (1024.0 ** 3)
                    co2_df['P_ram_W'] = co2_df['Used_Mem_GB'] * RAM_W_PER_GB
                    delta_T = (co2_df['Temp_C'] - T_AMBIENT_C).clip(lower=0)
                    co2_df['P_thermal_W'] = K_THERM_W_PER_C * delta_T
                    co2_df['P_total_W'] = co2_df['P_cpu_W'] + co2_df['P_ram_W'] + co2_df['P_thermal_W']
                    co2_df = co2_df.sort_values('time_sec')
                    co2_df['delta_t_sec'] = co2_df['time_sec'].diff().fillna(0.0)
                    co2_df['energy_kWh'] = (co2_df['P_total_W'] / 1000.0) * (co2_df['delta_t_sec'] / 3600.0)
                    co2_df['CO2_kg'] = co2_df['energy_kWh'] * CI
                    co2_df['CO2_rate_g_per_s'] = (co2_df['CO2_kg'] * 1000.0) / co2_df['delta_t_sec'].replace(0, pd.NA)
                    return float(co2_df['CO2_rate_g_per_s'].dropna().mean()) if co2_df['CO2_rate_g_per_s'].notna().any() else float('nan')

                avg_co2 = compute_avg_co2_from_raw(rawg)
                if (avg_co2 != avg_co2) and not rdf.empty:  # NaN check
                    # Fallback from aggregated dataframe
                    if all(c in rdf.columns for c in ['timestamp','cpu_percent','memory_used_mb','temperature']):
                        df2 = rdf[['timestamp','cpu_percent','memory_used_mb','temperature']].copy()
                        if pd.api.types.is_datetime64_any_dtype(df2['timestamp']):
                            t0 = df2['timestamp'].min(); df2['time_sec'] = (df2['timestamp'] - t0).dt.total_seconds()
                        else:
                            ts = pd.to_numeric(df2['timestamp'], errors='coerce'); df2['time_sec'] = (ts - ts.min()) / (1e6 if (ts.max()-ts.min())>1e6 else 1.0)
                        P_IDLE_W, P_MAX_W = 1.2, 8.0; T_AMBIENT_C = 25.0; K_THERM_W_PER_C = 0.2; RAM_W_PER_GB = 0.375; CI = 0.708
                        df2['U'] = pd.to_numeric(df2['cpu_percent'], errors='coerce')/100.0
                        df2['P_cpu_W'] = P_IDLE_W + (P_MAX_W - P_IDLE_W) * df2['U'].fillna(0.0)
                        df2['Used_Mem_GB'] = pd.to_numeric(df2['memory_used_mb'], errors='coerce').fillna(0.0)/1024.0
                        df2['P_ram_W'] = df2['Used_Mem_GB'] * RAM_W_PER_GB
                        delta_T = (pd.to_numeric(df2['temperature'], errors='coerce').fillna(T_AMBIENT_C) - T_AMBIENT_C).clip(lower=0)
                        df2['P_thermal_W'] = K_THERM_W_PER_C * delta_T
                        df2['P_total_W'] = df2['P_cpu_W'] + df2['P_ram_W'] + df2['P_thermal_W']
                        df2 = df2.sort_values('time_sec')
                        df2['delta_t_sec'] = df2['time_sec'].diff().fillna(0.0)
                        df2['energy_kWh'] = (df2['P_total_W']/1000.0) * (df2['delta_t_sec']/3600.0)
                        df2['CO2_kg'] = df2['energy_kWh'] * CI
                        df2['CO2_rate_g_per_s'] = (df2['CO2_kg'] * 1000.0) / df2['delta_t_sec'].replace(0, pd.NA)
                        avg_co2 = float(df2['CO2_rate_g_per_s'].dropna().mean()) if df2['CO2_rate_g_per_s'].notna().any() else float('nan')

                # Evaluation metrics means
                def get_mean(d, key):
                    return d.get('metrics', {}).get(key, {}).get('mean', float('nan'))
                f1_mean = get_mean(edata, 'f1')
                rougeL_mean = get_mean(edata, 'rougeL_f1')
                bert_mean = get_mean(edata, 'bertscore_f1')
                rows.append({
                    'model': model,
                    'task': task_display,
                    'avg_CO2_gps': avg_co2,
                    'F1': f1_mean,
                    'ROUGE_L_F1': rougeL_mean,
                    'BERT_F1': bert_mean,
                })
            cmp_df = pd.DataFrame(rows)
            if cmp_df.empty:
                st.info("No comparison data computed.")
            else:
                # Apply model-specific inference scaling to CO2 (user-provided factors)
                inference_factors = {
                    'phi': 0.3133,
                    'gemma': 0.410,
                    'qwen': 0.29,
                    'llama': 0.366,
                }
                cmp_df['adj_CO2_gps'] = cmp_df.apply(
                    lambda r: r['avg_CO2_gps'] * inference_factors.get(str(r['model']).lower(), 1.0)
                    if pd.notna(r['avg_CO2_gps']) else float('nan'), axis=1
                )
                # Normalize CO2 (lower is better) over valid rows only
                valid = cmp_df['adj_CO2_gps'].notna()
                if valid.any():
                    vmin = cmp_df.loc[valid, 'adj_CO2_gps'].min()
                    vmax = cmp_df.loc[valid, 'adj_CO2_gps'].max()
                    if pd.notna(vmin) and pd.notna(vmax) and vmax != vmin:
                        cmp_df.loc[valid, 'CO2_norm'] = (cmp_df.loc[valid, 'adj_CO2_gps'] - vmin) / (vmax - vmin)
                        cmp_df.loc[valid, 'CarbonScore'] = 1.0 - cmp_df.loc[valid, 'CO2_norm']
                    else:
                        cmp_df.loc[valid, 'CarbonScore'] = 1.0
                # Missing CO2 -> penalize to 0 CarbonScore
                cmp_df.loc[~valid, 'CarbonScore'] = 0.0

                # Final score: 0.6 Carbon + 0.25 BERT + 0.15 ROUGE-L (drop F1)
                rouge_col = cmp_df['ROUGE_L_F1'] if 'ROUGE_L_F1' in cmp_df.columns else pd.Series(0, index=cmp_df.index)
                cmp_df['FinalScore'] = 0.6*cmp_df['CarbonScore'] + 0.25*cmp_df['BERT_F1'].fillna(0) + 0.15*rouge_col.fillna(0)
                ranked = cmp_df[['model','task','avg_CO2_gps','adj_CO2_gps','ROUGE_L_F1','BERT_F1','FinalScore']].sort_values('FinalScore', ascending=False)
                st.dataframe(ranked)
                # Final result visualization (hide CarbonScore, still computed)
                fig_rank = px.bar(ranked, x='model', y='FinalScore', color='FinalScore',
                                  title='FinalScore by Model',
                                  hover_data=['avg_CO2_gps','ROUGE_L_F1','BERT_F1'])
                st.plotly_chart(fig_rank, use_container_width=True)
