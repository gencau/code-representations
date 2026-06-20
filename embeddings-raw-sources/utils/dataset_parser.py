# utils/dataset_parsers.py
import re
from pathlib import Path
from utils.record import Record
from utils.github_utils import GitUtils
from utils.file_utils import count_python_sources

_DIFF_RE = re.compile(r"^diff --git a/([^\s]+)", re.M)

def _extract_files(patch: str) -> list[str]:
    return list(dict.fromkeys(_DIFF_RE.findall(patch)))

def parse_lca(record, *, repo_root: Path) -> Record:
    return Record(
        id                 = record["id"],
        repo_owner         = record["repo_owner"],
        repo_name          = record["repo_name"],
        base_sha           = record["base_sha"],
        bug_description    = record["issue_body"],
        diff               = record["diff"],
        changed_files      = record["changed_files"],
        repo_files_without_test_count = record["repo_files_without_tests_count"],
        location           = GitUtils.saveFromGitHub(
                                record,
                                record["repo_owner"],
                                record["repo_name"],
                                record["base_sha"],
                                repo_root=repo_root
                                )
    )

def parse_swe_verified(record, *, repo_root: Path) -> Record:
    owner, name = record["repo"].split("/", 1)

    # Repo layout convention: <repo_root>/<owner>__<name>__commithash
    repo_name = owner + "__" + name + "__" + record["base_commit"]
    repo_dir = repo_root / Path(repo_name)
    if not repo_dir.exists():
        raise FileNotFoundError(f"{repo_dir} not found – check --repo_roots")

    return Record(
        id=record["instance_id"],
        repo_owner=owner,
        repo_name=name,
        base_sha=record["base_commit"],
        bug_description=record["problem_statement"],
        diff=record["patch"],
        changed_files=_extract_files(record["patch"]),
        repo_files_without_test_count=count_python_sources(repo_dir),
        location=repo_dir,
    )
