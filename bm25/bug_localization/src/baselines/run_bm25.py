import csv
import os, sys
import shutil
import time
import sys
import subprocess
import ast
from pathlib import Path
import time

import hydra
from datasets import config
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from backbones.base_backbone import BaseBackbone
from configs.baseline_configs import BaselineConfig
from data_sources.base_data_source import BaseDataSource
from utils.bm25_utils import build_json_files


@hydra.main(version_base="1.1", config_path="../../configs/baselines", config_name="bm25")
def main(cfg: BaselineConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Insert the project root at the beginning of sys.path so it has priority
    sys.path.insert(0, project_root)

    backbone: BaseBackbone = hydra.utils.instantiate(cfg.backbone)
    data_src: BaseDataSource = hydra.utils.instantiate(cfg.data_source)
    recreate_index = cfg.reindex

    output_path = HydraConfig.get().run.dir
    os.makedirs(output_path, exist_ok=True)
    results_csv_path = os.path.join(output_path, "results.csv")

    # changed files is the ground truth
    for dp, repo_content, changed_files in data_src:

        repo_dir = cfg.index_location + "/" + dp['repo_owner'] + "_" + dp['repo_name'] + "_" + dp['base_sha']
        if not os.path.exists(repo_dir):
            print(f"Repo dir {repo_dir} does not exist, skipping...")
            continue
        ext = dp['changed_files_exts']
        extension_dict = ast.literal_eval(ext)

        if recreate_index:
            index_location = repo_dir
            # We need to first create the json representations of our corpus
            build_json_files(Path(index_location), index_location + "/json", tuple(extension_dict.keys()))

            start_time = time.time()
            # Indexing command
            command = [
                sys.executable,  # Uses the same Python interpreter running this script
                "-m",
                "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", index_location + "/json",
                "--index", index_location + "/indexes",
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw"
            ]
            # then we create the index
            try:
                subprocess.run(command, check=True)
                print("Indexing completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error during indexing: {e}")
            end_time = time.time()

            print(f"Created index for {dp['base_sha']} in {end_time - start_time} seconds")
            dp['indexing_time'] = (end_time - start_time)

        issue_description = dp['issue_title'] + '\n' + dp['issue_body']
        start_time = time.time()

        language = ast.literal_eval(dp['changed_files_exts'])
        results_dict = backbone.localize_bugs(issue_description, repo_content, repo_dir, next(iter(language)))
        end_time = time.time()
        dp.update(results_dict)
        dp['time_s'] = (end_time - start_time) * 1000000

        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(dp.keys())
            writer.writerow(dp.values())

# Make sure to load the python environment and the conda pyserini environment before running!
# Call like this: python run_baseline.py
if __name__ == '__main__':
    cache_dir = config.HF_DATASETS_CACHE
    shutil.rmtree(cache_dir, ignore_errors=True)
    load_dotenv()
    main()
