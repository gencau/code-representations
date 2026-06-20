# On the Role of Retrieval-Oriented Code Representations in Agentic Bug Localization
The task is: given an issue with the bug description and the repository code in the state where the issue is reproducible, identify the files within the project that need to be modified to address the reported bug.

## Install Dependencies
We provide dependencies for the pip dependency manager, so please run the following command to install all the required packages in each subdirectory:

pip install -r requirements.txt

## Datasets 
Data for both datasets is available in HuggingFace: 

- Long Code Arena (LCA): https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization
- SWE-bench Verified (SWE): https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified

## RQ1 Repository Representation and Traditional Retrievers

### BM25:
All scripts for BM25 are located under bm25.

- Follow instructions on how to install Pyserini: https://github.com/castorini/pyserini.
- Configure your running environment in bug_localization/configs/baselines/bm25.yaml.
- Run with src/baselines/run_bm25.py (for LCA) or src/baselines/run_bm25_swe.py (for SWE-bench Verified).
- Compute metrics: src/baselines/metrics/compute_metrics_bm25.py

### Embeddings Retrieval

#### For Raw-Sources retrieval
Scripts are located under embeddings-raw-sources.

The databases with all embeddings for LCA and SWE are located under databases/embeddings.

Example usage: 

rag_pipeline.py rag_pipeline.py --repo_path <repos_directory> --embed_location <directory where to store the embeddings database>

#### All other representations
Script are located under embeddings-all/.


