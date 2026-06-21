import shutil
import os
import re
import json
import subprocess
import traceback

from pathlib import Path
import ast
import networkx as nx
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from lib2to3.pgen2.parse import ParseError

# This is to handle python2 code
fixers = get_fixers_from_package('lib2to3.fixes')
refactorer = RefactoringTool(fixers)

def is_test_file(file_path: str):
    test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", file_path.lower()))
    return any(word in words for word in test_phrases)

def build_json_files(source_dir:Path, index_dir: str, exts=('.py',), skip_tests=True):
    
    # clear up any existing index at that location
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir)

    all_py = [
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    print(f"Found {len(all_py)} total file(s) with extension(s) {exts}.")

    # Filter to only keep python files
    if skip_tests:
        filtered = []
        for p in all_py:
            # e.g. "astropy/modeling/tests/test_core.py"
            rel = p.relative_to(source_dir).as_posix()
            if is_test_file(rel):
                # skip it
                continue
            filtered.append(p)
        code_files = filtered
    else:
        code_files = all_py

    print(f"After skip_tests={skip_tests}, keeping {len(code_files)} file(s) to index.")

    
    for doc_id, path in enumerate(code_files):
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            doc = {
                'id': str(doc_id),
                'contents': content,
                'path': str(path.relative_to(source_dir).as_posix())
            }
            outpath = Path(index_dir) / f'{doc_id}.json'
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
        except Exception as e:
            print(f"Could not process file {path}, error: {str(e)}")

    print(f"Indexed {len(code_files)} JSON files to {index_dir}")

def build_python_dependency_graph(source_dir:Path, exclude_tests=True):
    """
    Build a file-level import dependency graph by manually parsing AST,
    automatically detecting top-level roots (e.g., 'src', 'lib', etc.),
    and mapping absolute imports to the correct directory.
    """
    project_root = Path(source_dir).resolve()
    EXCLUDE_DIRS = {"test", "tests", "testing"}

    roots = [project_root]
    for sub in project_root.iterdir():
        if not sub.is_dir():
            continue
        # optionally skip test dirs
        if exclude_tests and sub.name in EXCLUDE_DIRS:
            continue
        # only add if it actually contains Python files
        if any(sub.rglob("*.py")):
            roots.append(sub)

    if exclude_tests:
        py_files = [
            p for p in project_root.rglob("*.py")
            if not any(part in EXCLUDE_DIRS for part in p.relative_to(project_root).parts)
        ]
    else:
        py_files = list(project_root.rglob("*.py"))

    G = nx.DiGraph()

    # Add all Python files as nodes
    for py_file in py_files:
        rel_path = py_file.relative_to(project_root).as_posix()
        G.add_node(rel_path)

    # Parse imports and add edges
    for py_file in py_files:
        try:
            rel_src = py_file.relative_to(project_root).as_posix()
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(rel_src))
        except SyntaxError as err:
            print(f"Syntax error in {rel_src}:{err.lineno}:{err.offset} - {err.msg!r}")
            if (err.text):
                print(err.text.rstrip())
                print(" " + str((err.offset - 1)) + "^")

            try:
                refactored = refactorer.refactor_string(source, name=rel_src)
            except ParseError as pe:
                print(f"Refactor parse error {rel_src} - {pe!r}")
                continue
            except Exception as e_ref:
                print(f"Refactor error: {rel_src} - {type(e_ref).__name__}: {e_ref}")
                traceback.print_exc()
                continue
            try:
                # Sometimes it's a py2 vs py3 error, try to fix
                tree = ast.parse(str(refactored), filename=rel_src)
            except Exception as e:
                # Skip files that fail to parse
                print(f"Could not parse {rel_src} after refactoring, got: {e!r}")
                continue
        except Exception as exc:
            # This can happen on files that point to files that don't exist (e.g. __init__.py)
            print(f"Got unexpected error while attempting to parse {rel_src}: {exc!r}")

        for node in ast.walk(tree):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                module_names = [base] + [f"{base}.{alias.name}" for alias in node.names] if base else [alias.name for alias in node.names]
            else:
                continue

            for mod in module_names:
                if not mod:
                    continue
                parts = mod.split(".")
                # Try each root for mapping
                for root in roots:
                    candidate_file = root.joinpath(*parts).with_suffix(".py")
                    rel_dst = candidate_file.relative_to(project_root).as_posix()
                    if candidate_file.exists():
                        G.add_edge(rel_src, str(rel_dst))
                        break
                    candidate_pkg = root.joinpath(*parts) / "__init__.py"
                    rel_dst = candidate_pkg.relative_to(project_root).as_posix()
                    if candidate_pkg.exists():
                        G.add_edge(rel_src, str(rel_dst))
                        break

    return G

def build_java_kt_dependency_graph(source_dir:Path):
    proc = subprocess.run(
        ['java','-jar','/Users/gen/workspace/long-context-experiments/bug_localization/src/baselines/utils/java_dependency_graph/callgraph-0.0.1-SNAPSHOT.jar', 
            source_dir],
        capture_output=True,
        text=True,
        check=True
    )

    # Get the JSON
    try:
        data = json.loads(proc.stdout)
    except Exception as e:
        print(f"Got error while extracting JSON from stdout: {e}")

    # Create networkx graph from JSON
    G = nx.DiGraph()
    for node_path, edges in data['graph'].items():
        G.add_node(node_path)
        for neighbor_path in edges['outEdges']:
            G.add_edge(node_path, neighbor_path)
    
    return G