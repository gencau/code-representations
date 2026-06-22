#!/usr/bin/env python3
"""
Export file summaries from ChromaDB to their original directory structure.

This script reads summaries from ChromaDB collections and writes each summary
to a file at the path specified by its directory/filename metadata. This recreates
the repository structure with summaries instead of source code.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import chromadb


def remove_path_overlap(directory: str, filename: str) -> tuple:
    """
    Remove overlapping path segments between directory and filename.
    
    Examples:
        directory="/Volumes/project/data", filename="data/file.py"
        -> ("/Volumes/project/data", "file.py")
        
        directory="/Volumes/project/data/subdir", filename="data/subdir/file.py"
        -> ("/Volumes/project/data/subdir", "file.py")
        
        directory="/Volumes/project", filename="src/file.py"
        -> ("/Volumes/project", "src/file.py")  # no overlap
    
    Args:
        directory: Base directory path
        filename: Filename that may contain directory components
        
    Returns:
        Tuple of (directory, cleaned_filename)
    """
    if not directory or not filename:
        return directory, filename
    
    # Split paths into components
    dir_parts = Path(directory).parts
    file_parts = Path(filename).parts
    
    # If filename has no directory components, return as-is
    if len(file_parts) == 1:
        return directory, filename
    
    # Find the longest overlap
    # Check if any suffix of directory matches a prefix of filename
    max_overlap = 0
    for i in range(len(dir_parts)):
        dir_suffix = dir_parts[i:]
        # Check if this suffix matches the start of filename parts
        if len(file_parts) >= len(dir_suffix):
            if file_parts[:len(dir_suffix)] == dir_suffix:
                max_overlap = len(dir_suffix)
    
    # Remove the overlapping parts from filename
    if max_overlap > 0:
        cleaned_file_parts = file_parts[max_overlap:]
        cleaned_filename = str(Path(*cleaned_file_parts)) if cleaned_file_parts else ""
        return directory, cleaned_filename
    
    return directory, filename


def get_all_collections(chroma_path: str) -> List[str]:
    """
    Get list of all collection names in the ChromaDB database.
    
    Args:
        chroma_path: Path to ChromaDB persistent storage
        
    Returns:
        List of collection names
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()

    collection_names = []
    for col in collections:
        try:
            # Try to access name directly (older versions)
            collection_names.append(col.name)
        except (AttributeError, NotImplementedError):
            return collections # v0.6.0+ API, already returns collection names

    return collection_names


def get_collection_summaries(
    chroma_path: str,
    collection_name: str
) -> List[Dict]:
    """
    Retrieve all summaries from a ChromaDB collection.
    
    Args:
        chroma_path: Path to ChromaDB persistent storage
        collection_name: Name of the collection containing summaries
        
    Returns:
        List of dictionaries containing id, summary text, and metadata
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    
    # Get all documents from the collection
    results = collection.get(
        include=["documents", "metadatas"]
    )
    
    summaries = []
    for i, doc_id in enumerate(results["ids"]):
        summary_text = results["documents"][i]
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        
        summaries.append({
            "id": doc_id,
            "text": summary_text,
            "metadata": metadata
        })
    
    return summaries


def write_summary_files(summaries: List[Dict], base_output_dir: str, verbose: bool = False, dataset_name: str = "lca") -> int:
    """
    Write summaries to files at their original directory/filename paths.
    
    Args:
        summaries: List of summary dictionaries with metadata
        base_output_dir: Base directory where files will be written
        verbose: Whether to print detailed output
        
    Returns:
        Number of files successfully written
    """
    base_path = Path(base_output_dir)
    files_written = 0
    errors = []
    
    for summary in summaries:
        metadata = summary["metadata"]
        
        # Extract directory and filename from metadata
        directory = metadata.get("directory", "")
        filename = metadata.get("filename", metadata.get("file", ""))
        
        if not filename:
            errors.append(f"No filename found for summary {summary['id']}")
            continue
        
        # Remove any overlapping path segments
        directory, filename = remove_path_overlap(directory, filename)
        
        if not filename:
            errors.append(f"Filename became empty after overlap removal for {summary['id']}")
            continue
        
        # Construct full output path
        # Remove leading slash from directory to make it relative
        rel_directory = Path(directory).relative_to("/Volumes/T9/lca/cloned_repos/") if dataset_name == "lca" else Path(directory).relative_to("/Volumes/T9/repos/")

        output_path = base_path / rel_directory / filename
        print(f"Writing summary for {summary['id']} to {output_path}")

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write summary to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary["text"])
            
            if verbose:
                print(f"  Wrote: {output_path}")
            
            files_written += 1
            
        except Exception as e:
            errors.append(f"Error writing {output_path}: {e}")
    
    # Print any errors at the end
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")
    
    return files_written


def main():
    parser = argparse.ArgumentParser(
        description="Export file summaries from ChromaDB to their original directory structure"
    )
    parser.add_argument(
        "--chroma_path",
        required=True,
        help="Path to ChromaDB persistent storage directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory where summary files will be written"
    )
    parser.add_argument(
        "--dataset_name",
        default="lca",
        help="Dataset name"
    )
    parser.add_argument(
        "--collection",
        help="Specific collection name (optional, processes all collections if not specified)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each file written"
    )
    
    args = parser.parse_args()
    
    # Determine which collections to process
    if args.collection:
        collections = [args.collection]
        print(f"Processing collection: {args.collection}")
    else:
        collections = get_all_collections(args.chroma_path)
        print(f"Found {len(collections)} collections: {', '.join(collections)}")
    
    total_files = 0
    
    # Process each collection
    for collection_name in collections:
        print(f"\nProcessing collection: {collection_name}")
        
        try:
            # Retrieve summaries from ChromaDB
            summaries = get_collection_summaries(args.chroma_path, collection_name)
            print(f"  Retrieved {len(summaries)} summaries")
            
            if len(summaries) == 0:
                print("  Skipping empty collection")
                continue
            
            # Write summaries to files
            files_written = write_summary_files(summaries, args.output_dir, args.verbose, args.dataset_name)
            print(f"  Wrote {files_written} files")
            
            total_files += files_written
            
        except Exception as e:
            print(f"  Error processing collection {collection_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Total: Wrote {total_files} summary files to {args.output_dir}")
    print(f"{'='*60}")
    
    if not args.verbose:
        print("\nTip: Use --verbose to see each file being written")


if __name__ == "__main__":
    main()