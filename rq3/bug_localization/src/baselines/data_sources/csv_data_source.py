import os
from pathlib import Path
import sys
import pandas as pd

from src.baselines.data_sources.base_data_source import BaseDataSource

class CSVDataSource(BaseDataSource):

    def __init__(
            self,
            path: str
    ):
        self._path = path

    def __iter__(self):
        if Path(self._path).exists():
            df = pd.DataFrame(pd.read_csv(self._path))
        else:
            print(f"Couldn't find dataset {self._path}")
            sys.exit(1)
            
        for _,dp in df.iterrows():
            try:
                # Get repo content   
                if "final_files" not in dp:
                    repo_content = dp['topk_files']              
                else:
                    repo_content = dp['final_files']
                print(f"Got line {repo_content}")

                if "changed_files" not in dp:
                    changed_files = dp['expected_files']
                else:
                    # Get changed files between commits
                    changed_files = dp['changed_files']

                yield dp, repo_content, changed_files
            except Exception as e:
                print(f"Failed to get repo content for {self._path}", e)
                continue
