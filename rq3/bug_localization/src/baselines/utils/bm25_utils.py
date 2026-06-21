import shutil
import os
import json

from pathlib import Path

def build_json_files(source_dir:Path, index_dir: str, exts=('.py',)):
    
    # clear up any existing index at that location
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir)

    # Filter to only keep python files
    code_files = [p for p in source_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts]
    
    print(f"Found {len(code_files)} python files.")
    
    for doc_id, path in enumerate(code_files):
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            doc = {
                'id': str(doc_id),
                'contents': content,
                'path': str(path.relative_to(source_dir))
            }
            outpath = Path(index_dir) / f'{doc_id}.json'
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
        except Exception as e:
            print(f"Could not process file {path}, error: {str(e)}")

    print(f"Indexed {len(code_files)} JSON files to {index_dir}")