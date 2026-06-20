import re
import subprocess
from urllib.parse import urlparse, unquote
from pathlib import Path

# Find and return all links in a text
def extract_links(text) -> list:
    return re.findall(r'(https?://[^)\s]+)', text)

def extract_paths(text):
    return re.findall(r'\b[\w\/.-]+\.py\b', text)

def is_test_file(file_path: str):
    test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", file_path.lower()))
    return any(word in words for word in test_phrases)

# Extracts a path from a github url
def extract_file_path_from_url(url) -> str:
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.split('/')

    try:
        # Find the index of 'blob' which precedes the branch name
        blob_index = path_segments.index('blob')
        # The file path segments are after 'blob' and the branch name
        file_path_segments = path_segments[blob_index + 2:]
        # Join the segments to form the relative file path
        relative_file_path = '/'.join(file_path_segments)
        # Decode any URL-encoded characters
        relative_file_path = unquote(relative_file_path)
        return relative_file_path
    except ValueError:
        # path does not match expected, return empty string
        return ""

def get_path_within_repo(repo_base_path: str, filepath: str) -> str:
    """
    Returns a path minus the path to the repository itself.
    """
    repo = Path(repo_base_path)
    file = Path(filepath)
    
    try:
        return str(file.relative_to(repo))
    except ValueError:
        raise ValueError(f"File '{file}' is not in the repository '{repo}'")
    
def count_python_sources(path: str):
    repo_path = Path(path)

    if not repo_path.exists():
        print(f"No repository at {repo_path}, exiting.")
        return 0
    
    cmd = (
        f"find {repo_path} -type f -name '*.py' "
        "! -iname 'test_*.py' ! -iname '*_test.py' "
        "! -path '*/tests/*' ! -path '*/test/*' | wc -l"
    )
    result = subprocess.run(cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                        )
    num_files = int(result.stdout.strip() or 0)
    print(f"Got {num_files} files in repo {repo_path}")
    
    return num_files