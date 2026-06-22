#!/usr/bin/env python3
"""
Re-embed file summaries from ChromaDB with a different embedding model.

This script reads summaries from ChromaDB collections and creates new embeddings
using a specified model, storing them in a new ChromaDB database while preserving
collection names and metadata.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import shutil
from chromadb.config import Settings


class Storage:
    def __init__(self, path, collection_name):
        self.chroma_client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.collection = None
    
    def create_collection(self):
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}, 
            get_or_create=True
        )
    
    def set_collection(self, collection_name=None):
        """Set collection, optionally changing to a different collection."""
        if collection_name:
            self.collection_name = collection_name
        self.collection = self.chroma_client.get_collection(self.collection_name)
        
    def collection_exists(self, collection_name=None) -> bool:
        """Check if collection exists, optionally for a different collection name."""
        check_name = collection_name if collection_name else self.collection_name
        try:
            collection = self.chroma_client.get_collection(check_name)
            print(f"Collection '{check_name}' exists.")
            return True
        except:
            return False
    
    def get_chroma_client(self):
        return self.chroma_client
    
    def add(self, texts, embeddings, metadatas, ids):
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embeddings, top_k=5):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k, 
            include=["documents", "metadatas", "distances"]
        )
    
    def similarity_search_with_score(self, query_embedding, k=5, filter=None):
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k, 
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter:
            query_params["where"] = filter
        
        results = self.collection.query(**query_params)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        combined_results = list(zip(documents, metadatas, distances))
        
        return combined_results
    
    def cleanup(self):
        self.chroma_client.reset()
        shutil.rmtree("chroma_db")


def get_all_collections(chroma_path: str) -> List[str]:
    """
    Get list of all collection names in the ChromaDB database.
    
    Args:
        chroma_path: Path to ChromaDB persistent storage
        
    Returns:
        List of collection names
    """
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=True))
    collections = client.list_collections()

    collection_names = []
    for col in collections:
        try:
            collection_names.append(col.name)
        except (AttributeError, NotImplementedError):
            # For v0.6.0+, list_collections already returns collection names
            return collections

    return collection_names


def get_collection_summaries(
    storage: Storage
) -> List[Dict]:
    """
    Retrieve all summaries from a ChromaDB collection.
    
    Args:
        storage: Storage instance for the collection
        
    Returns:
        List of dictionaries containing id, summary text, and metadata
    """
    # Get all documents from the collection
    results = storage.collection.get(
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


def embed_and_store_summaries(
    summaries: List[Dict],
    output_storage: Storage,
    collection_name: str,
    model: SentenceTransformer,
    batch_size: int = 8
) -> int:
    
    if len(summaries) == 0:
        return 0
    
    if output_storage.collection_exists(collection_name):
        print(f"  Collection exists, skipping")
        return 0
    
    # Prepare data
    ids = [s["id"] for s in summaries]
    texts = [s["text"] for s in summaries]
    metadatas = [s["metadata"] for s in summaries]
    
    print(f"  Creating embeddings for {len(texts)} summaries...")
    print(f"  Model: {model}")
    print(f"  Max seq length: {model.max_seq_length} tokens")
    
    # Calculate appropriate character limit based on model
    max_chars = model.max_seq_length * 4  # ~4 chars per token
    print(f"  Truncating texts to {max_chars} characters")

    max_chars = min(max_chars, 8000)
    print(f"  Truncating texts to {max_chars} characters (for stability)")

    # Create embeddings
    all_embeddings = []
    failed_indices = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Truncate based on model's capabilities
        batch_texts = [t[:max_chars] if len(t) > max_chars else t for t in batch_texts]
        
        try:
            batch_embs = model.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embs.tolist())
            
        except Exception as e:
            print(f"\n  Batch failed: {e}")
            # Try individually
            for j, text in enumerate(batch_texts):
                try:
                    emb = model.encode([text], convert_to_numpy=True)
                    all_embeddings.append(emb[0].tolist())
                except:
                    failed_indices.append(i + j)
                    # Add zero vector or skip
    
    # Remove failed items
    if failed_indices:
        for idx in sorted(failed_indices, reverse=True):
            del all_embeddings[idx]
            del texts[idx]
            del metadatas[idx]
            del ids[idx]
    
    # Create collection
    output_storage.collection_name = collection_name
    output_storage.create_collection()
    
    # Add in chunks
    chunk_size = 1000
    for i in range(0, len(all_embeddings), chunk_size):
        end = min(i + chunk_size, len(all_embeddings))
        output_storage.add(
            texts=texts[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
    
    return len(all_embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed file summaries from ChromaDB with a different embedding model"
    )
    parser.add_argument(
        "--source-chroma-path",
        required=True,
        help="Path to source ChromaDB database with original summaries"
    )
    parser.add_argument(
        "--output-chroma-path",
        required=True,
        help="Path to output ChromaDB database for new embeddings"
    )
    parser.add_argument(
        "--collection",
        help="Specific collection name to process (optional, processes all if not specified)"
    )
    parser.add_argument(
        "--model",
        default="thenlper/gte-large",
        help="Embedding model to use (default: thenlper/gte-large)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Load embedding model
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    print(f"✓ Model loaded")
    
    # Determine which collections to process
    if args.collection:
        collections = [args.collection]
        print(f"\nProcessing collection: {args.collection}")
    else:
        collections = get_all_collections(args.source_chroma_path)
        print(f"\nFound {len(collections)} collections: {', '.join(collections)}")
    
    # Create Storage instances ONCE for source and output
    # Use a dummy collection name initially, we'll switch as needed
    print(f"\nInitializing source database: {args.source_chroma_path}")
    source_storage = Storage(args.source_chroma_path, collections[0] if collections else "dummy")
    
    print(f"Initializing output database: {args.output_chroma_path}")
    output_storage = Storage(args.output_chroma_path, "dummy")
    
    total_embeddings = 0
    
    # Process each collection
    for collection_name in collections:
        print(f"\n{'='*60}")
        print(f"Processing collection: {collection_name}")
        print('='*60)
        
        try:
            # Switch to source collection
            source_storage.set_collection(collection_name)
            
            # Retrieve summaries from source ChromaDB
            summaries = get_collection_summaries(source_storage)
            print(f"  Retrieved {len(summaries)} summaries")
            
            if len(summaries) == 0:
                print("  Skipping empty collection")
                continue
            
            # Embed and store in output ChromaDB
            start_time = time.time()
            count = embed_and_store_summaries(
                summaries=summaries,
                output_storage=output_storage,
                collection_name=collection_name,
                model=model,
                batch_size=args.batch_size
            )
            end_time = time.time()
            print(f"  Created {count} embeddings in {end_time - start_time:.2f} seconds")
            
            total_embeddings += count
            
        except Exception as e:
            print(f"  ✗ Error processing collection {collection_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Created {total_embeddings} embeddings across {len(collections)} collections")
    print(f"  Source: {args.source_chroma_path}")
    print(f"  Output: {args.output_chroma_path}")
    print("  Using cosine similarity for search")
    print('='*60)


if __name__ == "__main__":
    main()