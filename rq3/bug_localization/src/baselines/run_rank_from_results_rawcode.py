"""
Re-rank an existing set of candidate files using a backbone ranker.

This script takes a previously generated results CSV (e.g., from a retrieval
baseline) and applies a backbone's ranking step to reorder the ``final_files``
column for each issue.  The re-ranked results are written to
``results-ranking.csv`` inside the Hydra output directory.

Output CSV
----------
The output file ``results-ranking.csv`` contains all original columns plus any
fields returned by the backbone's ``_rank_results`` method (e.g., ranked file
order, scores) and ``time_s`` (elapsed time in microseconds per issue).
"""

import ast
import csv
import os, sys
import time
import sys

from src.baselines.data_sources.csv_data_source import CSVDataSource
from dotenv import load_dotenv
import hydra
from hydra.core.hydra_config import HydraConfig

from backbones.base_backbone import BaseBackbone
from data_sources.base_data_source import BaseDataSource
from configs.baseline_configs import BaselineConfig

@hydra.main(version_base="1.1", config_path="../../configs/baselines", config_name="rank_from_results_rawcode_swe")
def main(cfg: BaselineConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Insert the project root at the beginning of sys.path so it has priority
    sys.path.insert(0, project_root)

    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)
    results_data_src = hydra.utils.instantiate(cfg.data_source_results)
    topk: int = cfg.topk

    output_path = HydraConfig.get().run.dir
    os.makedirs(output_path, exist_ok=True)
    results_csv_path = os.path.join(output_path, "results.csv")

    # Redirect output to file
    log_file = open(os.path.join(output_path, 'output.log'), 'w')
    sys.stdout = log_file

      # Load already-processed IDs to support resuming interrupted runs
    processed_ids = set()
    if os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0:
        with open(results_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_id = row.get('id') or row.get('instance_id', '')
                if row_id:
                    processed_ids.add(str(row_id))
        print(f"Resuming: {len(processed_ids)} rows already processed.")

    # Pre-load results into a dict keyed by id to avoid positional misalignment
    # (HFDataSource silently skips failed repos via continue, which would desync a zip)
    results_by_id = {}
    for result_dp, _, _ in results_data_src:
        key = result_dp.get("index") if "index" in result_dp else result_dp.get("id")
        if key is not None:
            results_by_id[key] = result_dp
        else:
            key = result_dp.get("instance_id")
            results_by_id[key] = result_dp

    for dp, repo_content, changed_files in data_src:
        # Skip already-processed rows
        row_id = str(dp.get('id') or dp.get('instance_id', ''))
        if row_id in processed_ids:
            print(f"Skipping already-processed row: {row_id}")
            continue

        result_dp = results_by_id.get((row_id))
        if result_dp is None:
            print(f"No results found for issue {row_id}, skipping.")
            continue

        dp = dict(dp)  # HF dataset rows are read-only; need a mutable copy

        if "bug_description" in dp:
            issue_description = dp["bug_description"]
        elif "problem_statement" in dp:
            issue_description = dp["problem_statement"]
        else:
            issue_description = dp["issue_title"] + "\n" + dp["issue_body"]

        if "final_files" in result_dp:
            files = ast.literal_eval(result_dp["final_files"])
        else:
            files = ast.literal_eval(result_dp["topk_files"])
        file_list = files if isinstance(files, list) else []
        file_list = file_list[:topk]  # limit to top-k files for ranking step

        print(f"Processing issue: {row_id} with {len(file_list)} files to rank.")
        if file_list == []:
            print(f"No files to rank for issue {row_id}, skipping backbone ranking.")
            dp['ranked_files'] = []
            dp['rank_metadata'] = []
            dp['changed_files'] = changed_files
            dp['num_files_viewed'] = 0
            dp['time_s'] = 0
        else:
            db_key_field = 'base_commit'
            if 'db_key' in dp:
                db_key_field = 'db_key'
            elif 'base_sha' in dp:
                db_key_field = 'base_sha'
            start_time = time.time()
            results_dict = backbone.localize_bugs(issue_description, file_list, repo_content, dp[db_key_field])
            end_time = time.time()
            dp.update(results_dict)
            dp['changed_files'] = changed_files
            dp['time_s'] = (end_time - start_time) * 1000000

        write_header = not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(dp.keys())
            writer.writerow(dp.values())

    log_file.close()

if __name__ == '__main__':
    load_dotenv()
    main()