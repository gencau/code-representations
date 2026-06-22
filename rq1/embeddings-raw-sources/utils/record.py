# utils/records.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Record:
    id: str
    repo_owner: str
    repo_name: str
    base_sha: str
    location: Path                 # path on disk where the repo was cloned
    bug_description: str
    diff: str
    changed_files: List[str] = field(default_factory=list)
    repo_files_without_test_count: Optional[int] = None
