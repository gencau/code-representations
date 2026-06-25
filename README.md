# On the Role of Retrieval-Oriented Code Representations in Agentic Bug Localization

> **Task:** Given a bug report and a repository in the state where the issue is reproducible, identify the files that need to be modified to address the reported bug.

---

## Table of Contents

- [Setup](#setup)
- [Datasets, Results & Generated Databases](#datasets-results--generated-databases)
- [RQ1 — Traditional Retrievers](#rq1--repository-representation-and-traditional-retrievers)
- [RQ2 — LLM-based Retrieval](#rq2--llm-based-retrieval)
- [RQ3 — Post-retrieval Ranking](#rq3--post-retrieval-ranking)
- [Combining Results](#combining-results)
- [Analysis & Utilities](#analysis--utilities)

---

## Setup

Install dependencies for each subdirectory using:

```bash
pip install -r requirements.txt
```

---

## Datasets, results & generated databases

Both datasets are available on HuggingFace:

| Dataset | Link |
|---|---|
| Long Code Arena (LCA) | [JetBrains-Research/lca-bug-localization](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization) |
| SWE-bench Verified (SWE) | [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |


### Ready-to-Use Data
Alternatively, you will also find all repositories checked out at each task's commit SHA, all generated databases and all results [here](https://huggingface.co/buckets/gencau/code-representations) (>200GB of data, but takes several days to generate):
- lca_rawsources.zip: Long Code Arena raw source files.
- swe_rawsources.zip: SWE-bench Verified raw source files.
- all_results.zip: all our results, organized by RQ.
- BM25-summaries.zip: All repositories with raw source files replaced with their generated summaries, with BM25 indexes.
- chromadb*.zip: ChromaDB databases from raw source files, for each studied dense embedding.
- File paths: All dense embeddings generated for file paths (project structure).
- Summaries: All dense embeddings generated for all summary types.

Note the following mapping between paper/data summaries:
- prompt2: role-aware summary
- prompt3: detailed technical summary
- queries: generated bug reports

---

## RQ1 — Repository Representation and Traditional Retrievers

### BM25

Scripts are located under `bm25/`. Requires [Pyserini](https://github.com/castorini/pyserini).

1. Install Pyserini following the instructions at the link above.
2. Configure your environment in `bug_localization/configs/baselines/bm25.yaml`.
3. Run retrieval:
   - LCA: `python src/baselines/run_bm25.py`
   - SWE: `python src/baselines/run_bm25_swe.py`
4. Compute metrics: `python src/baselines/metrics/compute_metrics_bm25.py`

---

### Embedding Retrieval

#### Raw Sources

Scripts are located under `embeddings-raw-sources/`. Pre-built embedding databases for LCA and SWE are available under `databases/embeddings/`.

```bash
# LCA
python rag_pipeline.py --datasets=lca --repo_paths=<path to lca sources> --embed_location=<path to embeddings>

# SWE
python rag_pipeline.py --datasets=swe_verified --dataset_paths=<path to parquet file> --repo_path=<path to swe sources> --embed_location=<path to embeddings>
```

#### All Other Representations

Scripts are located under `embeddings-all/`.

**Adding summaries for an embedding model** (models loaded from HuggingFace):

```bash
# 1. Generate summaries and embeddings
python rag_pipeline.py --datasets=lca --repo_paths=<path to repo>
# → Results saved to results/ with a timestamp

# 2. Compute metrics
python compute_metrics.py --input=results/<results_file>.csv

# 3. Reuse summaries for a new embedding model
python summaries_to_embeddings.py \
  --source-chroma-path <database with summaries> \
  --output-chroma-path <output path> \
  --model <huggingface model name> \
  --batch_size <batch size>   # default 32; use 8 for Qwen3

# 4. Run pipeline (no re-indexing)
python rag_pipeline.py --datasets=lca --repo_paths=<path to repo> --no_add

# 5. Compute metrics
python compute_metrics.py --input=results/<results_file>.csv
```

**Extracting summaries to files for BM25:**

```bash
# 1. Export summaries
python summaries_to_file.py \
  --dataset_name <lca or swe> \
  --chroma_path <database with summaries> \
  --output_dir <output directory>

# 2. Edit bm25.yaml: set index_location and run_suffix

# 3. Run BM25 and compute metrics
python run_bm25.py
python metrics/compute_metrics_bm25.py --input <results file> --topk=5
```

---

## RQ2 — LLM-based Retrieval

Our three models use a 16K context window and can be recreated in Ollama using the modelfiles under `bug_localization/src/modelfiles/`:

```bash
ollama create -f <path to modelfile>
```

### File Paths Representation

Scripts are located under `rq2/file-paths/`.

```bash
# Configure
#   LCA: bug_localization/config/qwen2.3-coder.yaml
#   SWE: bug_localization/config/swe-bench-chat.yaml

# Run
python src/baselines/run_baselines.py      # LCA
python src/baselines/run_baseline_swe.py   # SWE
```

### Summaries Representation

Scripts are located under `rq2/summaries/`.

```bash
# Configure
#   LCA: bug_localization/config/qwen-agent.yaml
#   SWE: bug_localization/config/swe-agent.yaml

# Run
python src/baselines/run_agents.py       # LCA
python src/baselines/run_agents_swe.py   # SWE
```

Compute results with `src/baselines/metrics/compute_metrics_llm.py`.

---

## RQ3 — Post-retrieval Ranking

Scripts are located under `rq3/`.

### File Paths, Summaries, and Bug Report Representations

```bash
# Configure rank_from_results.yaml (LCA) or rank_from_results_swe.yaml (SWE):
#   backbone_type:    rank-w-filenames | rank-w-summaries | rank-w-bug-reports
#   prompt_target:    AgentContextPrompt   (filenames)
#                     AgentSummaryPrompt   (summaries)
#                     AgentBugSummaryPrompt (bug reports)
#   datasource_path:  path to results .csv file

python src/baselines/run_rank_from_results.py
python src/baseline/metrics/compute_metrics_ranked.py --max_k=5
```

### Raw Sources Representation

```bash
# Configure rank_from_results_rawcode.yaml (LCA) or rank_from_results_rawcode_swe.yaml (SWE):
#   backbone_type:   rank-w-sources
#   prompt_target:   AgentCodePrompt
#   datasource_results: path to results .csv file

python src/baselines/run_rank_from_results_rawcode.py
python src/baseline/metrics/compute_metrics_ranked.py --max_k=5
```

---

## Combining Results

Scripts are located under `rq2/raw-sources_summaries/`. Only RRF fusion is supported.

```bash
python src/baselines/combine_representations.py \
  --results_list=<comma-separated list of result files> \
  --topk=5 \
  --method=rrf \
  --output=<output path>
```

---

## Discussion

### UpSet Plot — Localized Files by Method

```bash
python utils/found_files_analysis.py \
  --data-paths=<comma-separated result files for a single retriever> \
  --topk=5
```

### Representation Footprint

```bash
# LCA example
python src/baselines/count_input_tokens.py \
  --source hf \
  --hub_name tiginamaria/bug-localization \
  --repos_dir <path to repos> \
  --configs py java kt \
  --split test \
  --output <output location>
```

### Indexing Times

```bash
python utils/summaries_to_embeddings.py \
  --collection=<unique collection ID> \
  --source-chroma-path <source database> \
  --output-chroma-path <output database> \
  --model <huggingface model name>
```