#!/usr/bin/env python3
"""
Count summary tokens by extracting them from ChromaDB.

Token counts use the openai/gpt-oss-20b tokenizer (o200k_harmony encoding).

Usage:
    # All collections
    python count_summary_tokens.py --chroma_path /path/to/chroma

    # Specific collection
    python count_summary_tokens.py --chroma_path /path/to/chroma --collection my_collection

    # Write per-item counts to a CSV
    python count_summary_tokens.py --chroma_path /path/to/chroma --output counts.csv
"""

import argparse
import csv
from typing import Dict, List
import os

import chromadb
from transformers import AutoTokenizer


def get_all_collections(chroma_path: str) -> List[str]:
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()

    collection_names = []
    for col in collections:
        try:
            collection_names.append(col.name)
        except (AttributeError, NotImplementedError):
            return collections  # v0.6.0+ API already returns names

    return collection_names


def get_collection_summaries(chroma_path: str, collection_name: str) -> List[Dict]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)

    results = collection.get(include=["documents", "metadatas"])

    summaries = []
    for i, doc_id in enumerate(results["ids"]):
        summary_text = results["documents"][i]
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        summaries.append({"id": doc_id, "text": summary_text, "metadata": metadata})

    return summaries


def count_summary_tokens(summaries: List[Dict], enc) -> List[Dict]:
    rows = []
    for summary in summaries:
        text = summary["text"] or ""
        token_count = len(enc.encode(text))
        rows.append({"doc_id": summary["id"], "summary_tokens": token_count})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Count summary tokens from ChromaDB.")
    parser.add_argument("--chroma_path", required=True, help="Path to ChromaDB persistent storage directory")
    parser.add_argument("--collection", help="Specific collection name (optional, processes all if not specified)")
    parser.add_argument("--output", default=None, help="Optional path to write per-item CSV")
    args = parser.parse_args()

    enc = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    if args.collection:
        collections = [args.collection]
        print(f"Processing collection: {args.collection}")
    else:
        collections = get_all_collections(args.chroma_path)
        print(f"Found {len(collections)} collections: {', '.join(collections)}")

    all_rows = []
    grand_total = 0
    collection_stats = []

    for collection_name in collections:
        print(f"\nProcessing collection: {collection_name}")
        try:
            summaries = get_collection_summaries(args.chroma_path, collection_name)
            print(f"  Retrieved {len(summaries)} summaries")
            if not summaries:
                print("  Skipping empty collection")
                continue

            rows = count_summary_tokens(summaries, enc)
            total = sum(r["summary_tokens"] for r in rows)
            grand_total += total

            for r in rows:
                r["collection"] = collection_name
            all_rows.extend(rows)
            collection_stats.append({"name": collection_name, "count": len(rows), "total": total})

        except Exception as e:
            print(f"  Error processing collection {collection_name}: {e}")
            continue

    if args.output and all_rows:
        os.makedirs(args.output, exist_ok=True)
        full_path = os.path.join(args.output, "results.csv")
        with open(full_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["collection", "doc_id", "summary_tokens"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nPer-item counts saved to '{args.output}'")

    n = len(all_rows)
    nc = len(collection_stats)
    print(f"\n=== Results (ChromaDB: {args.chroma_path}) ===")
    print(f"  {'Collection':<40}  {'Items':>8}  {'Total tokens':>14}  {'Avg tokens':>10}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*14}  {'-'*10}")
    for s in collection_stats:
        avg = s["total"] / s["count"]
        print(f"  {s['name']:<40}  {s['count']:>8,}  {s['total']:>14,}  {avg:>10,.0f}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*14}  {'-'*10}")
    if n > 0:
        print(f"  {'TOTAL':<40}  {n:>8,}  {grand_total:>14,}  {grand_total / n:>10,.0f}")
        print(f"\n  Collections                : {nc:,}")
        print(f"  Avg tokens / collection    : {grand_total / nc:,.0f}")
    else:
        print("  No data points found.")


if __name__ == "__main__":
    main()
