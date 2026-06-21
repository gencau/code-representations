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

Usage: 

rag_pipeline.py rag_pipeline.py --repo_path \<repos_directory\> --embed_location \<directory where to store the embeddings database\>

Example:
- For LCA: python rag_pipeline.py —datasets=lca —repo_paths=\<path to lca sources\> —embed_location=\<path to embeddings\>
- For SWE: python rag_pipeline.py --datasets=swe_verified --dataset_paths=\<path to parquet file\>  --repo_path=\<path to swe verified sources\> --embed_location=\<path to embeddings\>


#### All other representations
Script are located under embeddings-all/.

Process of adding summaries for an embedding model

We use models from hugging face.
- Set the path to where you want the database in rag_pipeline.py
- Set the model you want in record_processor
- Run: python rag_pipeline.py --datasets=lca --repo_paths=\<path to repo\>
- A results file is generated in the results folder with a timestamp.
- Run python compute_metrics.py --input=results/results_file_name.csv (use the one under llm-embeddings)

Once summaries are generated, we can reuse them from the database to generate new embeddings for other models:
- Run: python summaries_to_embeddings.py --source-chroma-path \<database that has the summaries\> —output-chroma-path \<path where to store the new embeddings\> --model \<hugging face model name\> —batch_size \<specify if the default 32 is too much. For Qwen3, 8 is best\>
- Run: python rag_pipeline.py —datasets=lca --repo_paths=\<path to repo\> —no_add
- A results file is generated in the results folder with a timestamp.
- Run python compute_metrics.py --input=results/results_file_name.csv

Extracting the summaries to file for BM25

- Run python summaries_to_file.py --dataset_name \<lca or swe\> —chroma_path \<database that has the summaries\> —output_dir \<path to where summaries will be written to files\> 
- From the bm25 project, edit the bm25.yaml file to have the index_location point to where the files were copied (root path). Edit the run_suffix to control the path where the results file will be stored (under the output folder).
- Run python run_bm25.py
- Run python metrics/compute_metrics_bm25.py —input \<path to results file\> —topk=5

## RQ2: LLM-based Retrieval

All scripts for raw sources and summary-based prompting are located under rq2/summaries.

All scripts for file paths prompting are located under rq2/file-paths.

Our 3 models using a 16K context window can be re-created in Ollama by using the configuration files located under bug_localization/src/modelfiles: ollama create -f \<path to modelfile\>

For file paths representation:
- Configure bug_localization/config/qwen2.3-coder.yaml for LCA, and swe-bench-chat.yaml for SWE
- Run src/baselines/run_baselines.py (LCA), run_baseline_swe.py (SWE)

For summaries:
- Configure bug_localization/config/qwen-agent.yaml (LCA) and swe-agent.yaml (SWE)
- Run src/baselines/run_agents.py (LCA) and src/baselines/run_agents_swe.py (SWE)

Compute results with src/baselines/metrics/compute_metrics_llm.py


## RQ3: Post-retrieval Ranking

All scripts are located under rq3.

Configure the yaml files for File paths and summaries representations:
- rank_from_results (LCA), rank_from_results_swe (SWE)
- Set the backbone type between: rank-w-filenames, rank-w-summaries, rank-w-bug-reports
- Set the retriever database location, for summaries
- Set the prompt target depending on the experiment type: 
    - AgentContextPrompt: rank-w-filenames
    - AgentSummaryPrompt: rank-w-summaries (any summary except bug report summaries)
    - AgentBugSummaryPrompt: rank-w-bug-reports

- Set the datasource path to the results .csv file you want to rank.
- Set the proper configuration file to use in src/baselines/run_rank_from_results.py

* Run: python src/baselines/run_rank_from_results.py
* Compute results by running src/baseline/metrics/compute_metrics_ranked.py with max_k=5.

Configure the yaml files for the raw sources representation:
- rank_from_results_rawcode (LCA), rank_from_results_rawcode_swe (SWE)
- Set the backbone type to rank-w-sources
- Set the prompt target to AgentCodePrompt
- Set the datasource_results path to the results .csv file you want to rank.
- Set the proper configuration file to use in src/baselines/run_rank_from_results_rawcode.py

* Run: python src/baselines/run_rank_from_results_rawcode.py
* Compute results with src/baseline/metrics/compute_metrics_ranked.py with max_k=5.

## Combining Results

Under rq2/raw-sources_summaries: combine_representations.py. Only rrf method is used.

Example:
python src/baselines/combine_representations.py --results_list=\<comma-separated list of different representations\> --topk=5 --method=rrf --output=\<output_path\>

## Analysis of localized files by different methods (for UpSet plot generation)
Scripts located under rq2/raw-sources_summaries.

Example:
python utils/found_files_analysis.py --data-paths=\<comma-separated list of different representations for a single retriever\> --topk=5


## Computing Representation Footprint
Scripts located under rq2/raw-sources_summaries.

Example for LCA:

python src/baselines/count_input_tokens.py --source hf --hub_name tiginamaria/bug-localization --repos_dir \<path to repos\> --configs py java kt --split test --output \<output location\>

## Computing Indexing Times
Scripts located under rq2/raw-sources_summaries.

python utils/summaries_to_embeddings.py --collection=\<unique collection ID\> --source-chroma-path \<path to source database\> --output-chroma-path \<path to output database\> --model \<model huggingface name\>

## Results
All results can be found as a separate package in Zenodo, due to their size.

## Databases
All generated chroma-db databases can be found as a separate package in Zenodo, due to their size.