from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import git
import re
import unidiff
from unidiff import PatchSet

from src.utils.file_utils import is_test_file


def get_changed_files_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str,
                                      extensions: Optional[list[str]] = None,
                                      ignore_tests: bool = False) -> List[str]:
    """
    Get changed files between `first_commit` and `second_commit`
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :param extensions: list of file extensions to get
    :return: list of changed files
    """

    pull_request_diff = get_diff_between_commits(repo_path, first_commit_sha, second_commit_sha)
    changed_files = parse_changed_files_from_diff(pull_request_diff)
    filtered_changed_files = []

    for changed_file in changed_files:
        if ignore_tests and is_test_file(changed_file):
            continue

        if extensions and any(changed_file.endswith(ext) for ext in extensions):
            filtered_changed_files.append(changed_file)

    return filtered_changed_files


def get_changed_files_in_commit(repo_path: str, commit_sha: str) -> List[str]:
    """
    Get changed files in commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: sha of commit
    :return: list of changed files
    """

    pull_request_diff = get_diff_commit(repo_path, commit_sha)
    return parse_changed_files_from_diff(pull_request_diff)


def get_changed_files_and_lines_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str) \
        -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    For each changed files get changed lines in commit
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :return: dict from file path to lines for each changed files according to diff
    """

    pull_request_diff = get_diff_between_commits(repo_path, first_commit_sha, second_commit_sha)
    return parse_changed_files_and_lines_from_diff(pull_request_diff)


def get_diff_between_commits(repo_path: str, first_commit_sha: str, second_commit_sha: str) -> str:
    """
    Get git diff between `first_commit` and `second_commit` https://matthew-brett.github.io/pydagogue/git_diff_dots.html
    :param repo_path: path to directory where repo is cloned
    :param first_commit_sha: sha of first commit
    :param second_commit_sha: sha of second commit
    :return: git diff in standard string format
    """

    repo = git.Repo(repo_path)

    return repo.git.diff("{}...{}".format(first_commit_sha, second_commit_sha))


def get_diff_commit(repo_path: str, commit_sha: str) -> str:
    """
    Get git diff for commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: sha of commit
    :return: git diff in standard string format
    """

    repo = git.Repo(repo_path)
    return repo.git.show(commit_sha)


TEST_DIR_NAMES = {"test", "tests", "testing"}

def _unidiff_files(diff_str: str) -> List[str]:
    """
    Return file paths from a git diff in the exact order they appear,
    excluding anything located in a test/tests/testing directory.
    Duplicates are removed while preserving first occurrence.
    """
    files_in_order: List[str] = []
    seen = set()

    try:
        # NOTE: splitlines(keepends=True) because PatchSet expects an
        # iterable of *lines*, not one big string.
        patch = unidiff.PatchSet(
            diff_str.splitlines(keepends=True),
            metadata_only=True,        # <- really honoured here
        )
    except Exception:
        patch = _regex_files(diff_str)

    for patched_file in patch:
        # Strip leading "a/" if present (Git diff format)
        path = patched_file.source_file.split("a/", 1)[-1]

        # Skip if any directory component is test/tests/testing (case-insensitive)
        if any(part.lower() in TEST_DIR_NAMES for part in path.split('/')):
            continue

        if path not in seen:
            seen.add(path)
            files_in_order.append(path)

    return files_in_order

_PATH_RE = re.compile(r'^\+\+\+\s+b/(.+)$', re.MULTILINE)

def _regex_files(diff: str) -> list[str]:
    return list(dict.fromkeys(_PATH_RE.findall(diff)))   # dedupe + preserve order

def parse_changed_files_from_diff(diff_str: str) -> list[str]:
    try:
        return _unidiff_files(diff_str)      # the function from Fix 1
    except:
        return _regex_files(diff_str)
    
def parse_added_files_from_diff(diff_str: str) -> List[str]:
    source_files = {
        patched_file.target_file.split("b/", 1)[-1]
        for patched_file in unidiff.PatchSet.from_string(diff_str) if patched_file.is_added_file
    }

    return list(source_files)


def parse_changed_files_and_lines_from_diff(diff_str: str) -> Dict[str, list[tuple[int, str, str]]]:
    """
    Parse change file names and lines in it from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: dict from file path to lines for each changed files according to diff
    """
    changed_files_and_lines = defaultdict(list)
    patch_set = unidiff.PatchSet(diff_str)
    for patched_file in patch_set:
        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    changed_files_and_lines[patched_file.path].append((line.target_line_no - 1, 'a', line.value))
                elif line.is_removed:
                    changed_files_and_lines[patched_file.path].append((line.source_line_no - 1, 'r', line.value))

    return dict(changed_files_and_lines)


def get_repo_content_on_commit(repo_path: str, commit_sha: str,
                               extensions: Optional[list[str]] = None,
                               ignore_tests: bool = False) -> Dict[str, Optional[str]]:
    """
    Get repo content on specific commit
    :param repo_path: path to directory where repo is cloned
    :param commit_sha: commit shat on what stage get repo's content
    :return: for all files in repo on`commit_sha` stage map from file path (relative from repo root) to it's content
    """
    repo = git.Repo(repo_path)
    commit = repo.commit(commit_sha)

    file_contents = {}
    for blob in commit.tree.traverse():
        if blob.type == "blob":
            file_path = str(blob.path)
            if extensions is not None and not any(file_path.endswith(ext) for ext in extensions):
                continue
            if ignore_tests and is_test_file(file_path):
                continue
            try:
                content = blob.data_stream.read().decode('utf-8', errors='replace')
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = None
    return file_contents
