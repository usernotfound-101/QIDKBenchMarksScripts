usernotfound-101:
# QIDK Edge AI Sustainability Benchmark Suite

This directory contains all the code used to run and analyze sustainability‑focused benchmarks for small language models (SLMs) and RAG (Retrieval‑Augmented Generation) workloads, primarily on an Android edge device via `adb` and `llama.cpp`.

The suite covers three main tasks:

1. **Q&A on SQuAD** (phone, multi‑model, JSON‑structured outputs)
2. **Article summarization** (phone, JSON‑structured outputs)
3. **Resume‑based RAG benchmark** with **carbon accounting** and an HTML dashboard

---

## 1. Q&A Benchmark (`generate_qna.py`)

**File:** [Code/generate_qna.py](Code/generate_qna.py)

Interactive CLI tool that benchmarks multiple SLMs on the SQuAD v1 validation set, running inference on a phone via `adb` and `llama.cpp`.

### Key features

- Loads SQuAD validation split with 🤗 `datasets`.
- Groups questions by shared context and samples a configurable number of **topics** and **questions per topic**.
- Runs on a selected model deployed on the phone:
  - Llama 3.2 (3B)
  - Phi 3.5 Mini
  - Gemma 2 (4B)
  - Qwen 2.5 (4B)
- Uses a **JSON schema** enforced by `--json-schema` in `llama-cli` to ensure structured outputs:
  - Output format:  
    ```json
    { "answers": [ { "id": "q0001", "answer": "..." }, ... ] }
    ```
- Stores all artifacts locally:
  - `local_prompts/<model_dir>/`: prompt text for each batch
  - `local_results/<model_dir>/`: raw per‑batch results and merged JSON
  - `local_logs/<model_dir>/`: stderr/runtime logs from device

### Typical workflow

1. Run:
   ```bash
   cd Code
   python3 generate_qna.py
   ```
2. Provide:
   - Number of topics
   - Max questions per topic
   - Model selection (1–4)
3. Script:
   - Prepares batches (`generate_batches`)
   - Builds batched prompts (`create_prompt_for_batch`)
   - Pushes prompts via `adb` (`push_batch_to_phone`)
   - Executes `llama-cli` with JSON schema (`run_model_on_phone`)
   - Pulls and merges batch results (`pull_results`, `merge_batch_results`)
   - Cleans up remote temp files (`cleanup_remote`)
4. At the end it prints:
   - Total number of answers, a few samples
   - Local paths for prompts, results, and logs

This script is the reference for **end‑to‑end Q&A benchmarking on device** with reproducible prompts and structured outputs.

---

## 2. Article Summarization Benchmark (`generate_summary.py`)

**File:** [Code/generate_summary.py](Code/generate_summary.py)  
**Supporting text files:**  
- [Code/articles.txt](Code/articles.txt) – raw articles, separated by `=== ARTICLE N ===` markers  
- [Code/highlights.txt](Code/highlights.txt) – gold/reference summaries, separated by `=== SUMMARY N ===` markers  
- [Code/a1.txt](Code/a1.txt) – additional article content (same marker style)

This script evaluates summarization quality of SLMs on device.

### Key features

- Interactive CLI (`get_user_config`):
  - Asks for `articles` file (default `articles.txt`)
  - Asks for `highlights` file (default `highlights.txt`)
  - Asks for output JSON filename (default `summaries_output.json`)
  - Lets user select one of the phone models (same list as Q&A)
- Input parsing:
  - `read_file_sections` splits articles and highlights using regex delimiters.
  - Returns `(article_number, content)` pairs.
- Phone inference path mirrors `generate_qna.py`:
  - `push_prompt_to_phone`:
    - Writes a summarization prompt including the full article:
      - “Summarize this article briefly in 3–5 sentences…”
    - Saves locally in `_temp_summary_workspace/`
    - Pushes to `/data/local/tmp` via `adb`.
  - `run_inference_on_phone`:
    - Builds `llama-cli` command in `LLAMA_BIN_DIR`.
    - Uses `--json-schema` enforcing:
      ```json
      { "summary": "..." }
      ```
    - Writes results to `summary_result_<n>.json` on device, then pulls to local temp.
- Robust parsing:
  - `parse_summary_output` tries:
    1. Direct JSON parse from raw file.
    2. Extracting a JSON sub‑string from noisy output.
    3. Regex fallback on `summary: '...'`.
    4. Last‑resort text cleanup.
- Output:
  - Final JSON (`summaries_output.json` by default) with entries:
    ```json
    {
      "article_number": "1",
      "highlight": "gold summary...",
      "generated_summary": "model summary...",
      "article_text": "...",
      "generation_logs": "...stderr or errors...",
      "raw_model_output": "raw or parsed text"
    }
    ```
- Cleans up local temp folder after run.

Run with:

```bash
cd Code
python3 generate_summary.py
```

---

## 3. SLM Evaluation on SQuAD with BERTScore (bert_bench.py)

**File:** bert_bench.py

Benchmarks multiple .gguf SLMs on SQuAD using a **host‑side** script that calls a local `brun-cli.sh` wrapper (for `llama.cpp` or similar).

### Key features

- Uses 🤗 SQuAD (`datasets.load_dataset("squad")`).
- Randomly samples `NUM_QUESTIONS` (default 10) with a fixed seed.
- For each question:
  - Builds a concise prompt:
    > Answer in EXACTLY 20 words or less. Be concise: \<question\>
  - Calls `brun-cli.sh` via `subprocess.run` with:
    - `M=<model_name>` and `D=<device>` env vars
    - `-no-cnv`, `-n 50` to cap tokens.
- Per‑model outputs:
  - Response directory `responses_<model_slug>/`
  - Log directory `logs_<model_slug>/`
  - Each question’s:
    - Context and question in `<hash>_question.json`
    - Raw response in `<hash>_response.txt`
    - CLI logs in `<hash>_log.txt`
- Response pruning:
  - `extract_response` trims repeated content, removes stop markers (e.g. `[end of text]`), and caps to ~25 words.
- Evaluation:
  - `evaluate_bert_scores` uses `bert-score` to compute **precision/recall/F1** per question and per model.
- Outputs:
  - `bert_scores.csv` – tabular summary
  - `detailed_results.json` – questions, model responses, and BERT scores.

Run with:

```bash
cd Code
python3 bert_bench.py
```

Requires:
- `datasets`
- `bert-score`
- `pandas`

---

## 4. RAG Benchmark & Dashboard (`RAG/`)

All RAG‑specific components live under RAG.

### 4.1. RAG Benchmark Script (rag_benchmark.py)

**File:** rag_benchmark.py  
**Spec / instructions:** instructions.md  
**Questions/answers dataset:** dataset.json

This script implements the full pipeline described in instructions.md for a **resume‑driven Q&A RAG benchmark** with carbon tracking.

#### High‑level flow

1. **Inputs**:
   - `--resume`: path to `resume.pdf` (or `.txt` / `.docx`)
   - `--questions`: path to `dataset.json` (10 Q/A pairs)
   - `--model`: `llama.cpp` model path (e.g. `gguf` file)
   - `--llama-cli`: path to `llama-cli` binary
   - `--output-dir`: root directory where run artifacts are stored
   - Retrieval config: `--chunk-size`, `--chunk-overlap`, `--top-k`
2. **Resume loading** (`load_file`):
   - Supports PDF (via `PyPDF2`), TXT, and DOCX (via `docx2txt`).
3. **Chunking & retrieval**:
   - `chunk_text`: overlapping windowing over raw resume text.
   - `RetrievalEngine`:
     - Builds `TfidfVectorizer` over chunks.
     - Uses cosine similarity for ranking.
     - `search(question)` returns top‑k chunks with scores and latency.
4. **Prompt construction**:
   - Model‑family specific templates: `PromptTemplate` and `PROMPT_TEMPLATES`.
   - `detect_model_family` infers family from model path (`"gemma"`, `"qwen"`, else `"llama"`).
   - Prompt always includes:
     - A `SYSTEM_PROMPT` enforcing:
       - Only use resume context.
       - Verbatim facts.
       - A delineated `***ANSWER***` section and context span.
   - `render_context` formats retrieved chunks with rank and similarity.
5. **Carbon tracking**:
   - `start_carbon_tracker` / `stop_carbon_tracker` wrap `codecarbon.EmissionsTracker` for:
     - Overall session (`rag_session_<slug>`)
     - Per‑question generation (`generation_q<id>_<slug>`)
   - Emissions recorded in kg CO₂e.
6. **Running `llama.cpp`**:
   - `build_llama_command` builds the CLI command (prompt via `-p`).
   - `run_single_example`:
     - Launches the process with `subprocess.Popen`.
     - Attaches a background `MemoryMonitor` that samples `/proc/<pid>/status`:
       - Peak RSS in KB and sample history.
     - Collects stdout and stderr; raises if exit code != 0.
7. **Perf metrics parsing**:
   - Regexes over stderr for:
     - `load time`, `prompt eval time`, `eval time`, `sampling time`, `total time`, `graphs reused`.
   - `parse_metrics` aggregates them; `compute_throughput` computes tokens/sec.
   - `summarize_memory_samples` -> min/avg/max memory in MB.
8. **Metrics & evaluation**:
   - Answer normalization & metrics:
     - `normalize_answer`, `tokenize`
     - `exact_match` (EM)
     - `safe_f1` and `best_f1` across multiple gold answers.
   - Semantic similarity:
     - `semantic_similarity` uses TF‑IDF and cosine between model answer and joined gold answers.
   - Carbon metrics per question:
     - `RetrievalResult` holds `latency_ms`, `carbon_kg`, and retrieved chunks.
     - Generation carbon from `CodeCarbon`.
9. **Aggregation**:
   - Raw outputs (per question) and parsed metrics are persisted.
   - `evaluate_entries` computes dataset‑level EM/F1 + carbon aggregates.
   - `SERIES_SPECS` + `summarize_series` and `summarize_entries` compute summary stats with min/max/avg/median/p90 for many metrics (latency, memory, carbon, tokens, EM/F1, etc.).
10. **Artifacts saved under** `<output_dir>/<model_slug>/`:
    - `raw_outputs_<slug>.json` – full stdout, stderr, metrics per question.
    - `parsed_outputs_<slug>.json` – cleaned view with key metrics and answers.
    - `raw_metrics.json` – perf/latency metrics derived from stderr.
    - `evaluation_metrics.json` – dataset‑level EM/F1 + carbon summary.
    - `summary_metrics.json` – statistical summary per metric (for plotting).
    - `prompt_log.json` – per‑question prompt and retrieval context details.
    - `summary.json` – condensed overview tying all above together.

Run example:

```bash
cd Code/RAG
python3 rag_benchmark.py \
  --resume ../resume.pdf \
  --questions dataset.json \
  --model /path/to/model.gguf \
  --llama-cli /path/to/llama-cli \
  --output-dir ./gemma3-4b-ragres2 \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --top-k 5
```

---

### 4.2. RAG Dashboard Builder (build_rag_dashboard_2.py)

**File:** build_rag_dashboard_2.py  
**Generated HTML example:** rag_dashboard_2.html

Converts multiple rag_benchmark.py runs into a **single interactive dashboard** that compares models across **accuracy, latency, throughput, retrieval behavior, and carbon emissions**.

#### Inputs

- A root directory holding multiple `*-ragres2` run folders, each with:
  - `summary.json`
  - `summary_metrics.json`
  - `parsed_outputs_*.json` (per‑question data)

#### ModelReport

Aggregated per‑run metrics are loaded into a `ModelReport` dataclass, including:

- Latencies:
  - `retrieval_latency_ms`
  - `generation_latency_ms`
  - `load_ms`
  - `total_latency_ms`
- Accuracy:
  - `semantic_similarity`
  - `f1`
  - `blended_accuracy` (combined EM + F1 + semantic)
- Carbon:
  - `retrieval_carbon_avg`
  - `generation_carbon_avg`
  - Derived properties:
    - `total_carbon_per_question`  
      $= \text{retrieval} + \text{generation}$
    - `generation_carbon_per_100_tokens`  
      $= \dfrac{\text{generation\_carbon\_avg}}{\text{eval\_tokens\_avg}} \times 100$
    - `carbon_ratio`  
      $= \dfrac{\text{retrieval\_carbon\_avg}}{\text{generation\_carbon\_avg}}$
- Retrieval behavior:
  - `avg_graphs_reused`
  - `avg_retrieved_chunks`
  - `avg_top_similarity`
  - `avg_retrieval_carbon_per_chunk`
- Answer length and throughput:
  - `answer_chars_avg`
  - `throughput_eval_tps`
  - `eval_tokens_avg`
- Memory:
  - `memory_mb`

Summary data per model is built via `collect_reports`, which reads summary metrics and aggregates additional data from parsed outputs (`_aggregate_parsed_metrics`).

#### Charts

Using Plotly, the script builds multiple figures (see `build_chart_sections`):

- **Per‑Question Carbon Split** (`build_carbon_split_chart`)
  - Stacked bars of retrieval vs generation carbon (kg CO₂e per question).
- **Accuracy vs Carbon** (`build_accuracy_vs_carbon_chart`)
  - Scatter: blended accuracy vs generation carbon.
- **Answer Length vs Generation Carbon** (`build_answer_length_vs_carbon_chart`)
  - Scatter: average answer chars vs generation carbon.
- **Latency vs Carbon** (`build_latency_vs_carbon_bubble`)
  - Bubble chart: total latency vs carbon (bubble size encodes retrieval latency).
- **Retrieval Latency vs Carbon** (`build_retrieval_latency_vs_carbon`)
- **Retrieval Depth vs Accuracy** (`build_retrieval_depth_vs_accuracy_chart`)
  - Avg retrieved chunks vs blended accuracy.
- **Throughput vs Carbon Intensity** (`build_throughput_vs_carbon`)
  - Tokens/sec vs carbon per 100 tokens.
- **Carbon per Chunk vs Similarity** (`build_carbon_per_chunk_vs_similarity_chart`)

Each chart has a human‑readable description generated by helper functions like `_describe_carbon_split`, `_describe_accuracy_vs_carbon`, `_describe_retrieval_latency_vs_carbon`, etc., which automatically reference best/worst models and concrete numeric values.

#### Quick Insights block

`build_insights` generates an HTML `<ul>` with bullets such as:

- Model with lowest generation carbon intensity (kg per 100 tokens).
- Model that leads on blended accuracy.
- Fastest end‑to‑end model.
- Lowest retrieval‑to‑generation carbon ratio.
- Lowest carbon per retrieved chunk.
- Most “concise and green” model (short answers + low carbon).

These insights are included at the top of the dashboard as a **Quick Insights** card.

#### Dashboard rendering

[`render_dashboard`](Code/RAG/build_rag_dashboard_2.py) uses Plotly’s `pio.to_html` to embed each chart in a responsive card layout with custom CSS (see rag_dashboard_2.html for an example).

Run:

```bash
cd Code/RAG
python3 build_rag_dashboard_2.py \
  --rag-root . \
  --output rag_dashboard_2.html
```

Open the resulting rag_dashboard_2.html in a browser.

---

## 5. Graphing CSV Metrics (graphs.py)

**File:** graphs.py

Utility script for plotting grouped metric graphs from a CSV export (e.g., from previous experiments).

### Behavior

- Reads a CSV (default path `~/Documents/run_qwen.csv`) into a pandas DataFrame.
- Cleans column names and strips whitespace in key columns (`Metric`, `Category`).
- Optionally filters out a specific category (e.g. `"DSP - Application"`).
- Prints:
  - DataFrame `head()`
  - DataFrame info
  - Unique categories and metrics after filtering
- Then generates grouped plots per subset of metrics using `matplotlib`.

You can adapt the CSV filename and the grouping logic to fit your own exports.

Run:

```bash
cd Code
python3 graphs.py
```

---

## 6. Python Dependencies (req.txt)

**File:** req.txt

Large environment file listing Python packages used across the RAG and benchmarking scripts, including:

- Core libraries: `numpy`, `pandas`, `scikit‑learn`, `matplotlib`, `plotly`
- NLP and model tooling: `transformers`, `tokenizers`, `huggingface-hub`
- Carbon tracking: `codecarbon` (implied via rag_benchmark.py)
- File parsing: `PyPDF2`, `docx2txt`
- Others: `requests`, `tqdm`, and many OS / desktop related libs.

For a minimal environment focused on the benchmark core, you mainly need:

- `numpy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `plotly`
- `datasets`
- `bert-score`
- `codecarbon`
- `PyPDF2`
- `docx2txt`

---

## 7. How the Pieces Fit Together

- **On‑device experiments**:
  - generate_qna.py and generate_summary.py are the primary tools for running **Q&A** and **summarization** benchmarks on the Android device, targeting multiple small LLMs via `llama-cli` and `adb`.
  - Both enforce **JSON schemas** to ensure robust parsing and reproducible evaluation.
- **Model‑centric eval on host**:
  - bert_bench.py focuses on **BERTScore** evaluation of multiple SLMs on SQuAD, driven by a local wrapper (`brun-cli.sh`).
- **RAG + Carbon + Dashboard**:
  - rag_benchmark.py adds **retrieval**, **per‑stage carbon accounting**, **EM/F1/semantic similarity**, and **rich metrics** on a resume Q&A workload.
  - build_rag_dashboard_2.py gathers the outputs of multiple `rag_benchmark` runs into a **single interactive dashboard** that allows you to visually trade off **accuracy vs carbon vs latency vs retrieval quality**.

Together, these scripts form a coherent **Edge AI Sustainability Benchmarking** toolkit for comparing small LLMs and RAG setups under realistic workloads and constraints.
