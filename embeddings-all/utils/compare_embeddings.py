#!/usr/bin/env python3
"""
Compare embeddings from two ChromaDB databases to verify they're different.
"""
import chromadb
import numpy as np
from pathlib import Path
from typing import List
from chromadb.config import Settings


def get_all_collections(client: chromadb.PersistentClient) -> List[str]:
    """
    Get list of all collection names in the ChromaDB database.
    
    Args:
        client: ChromaDB client
        
    Returns:
        List of collection names
    """
    collections = client.list_collections()
    collection_names = []
    for col in collections:
        try:
            collection_names.append(col.name)
        except (AttributeError, NotImplementedError):
            # For v0.6.0+, list_collections already returns collection names
            return collections
    return collection_names


def compare_embeddings(client1, client2, collection_name, sample_size=10):
    """Compare embeddings from two databases."""
    
    print(f"\nComparing collection: {collection_name}")
    
    try:
        col1 = client1.get_collection(name=collection_name)
        col2 = client2.get_collection(name=collection_name)
    except Exception as e:
        print(f"  ✗ Error loading collections: {e}")
        return False
    
    # Get embeddings from both
    result1 = col1.get(limit=sample_size, include=["embeddings", "documents"])
    result2 = col2.get(limit=sample_size, include=["embeddings", "documents"])
    
    if result1["embeddings"] is None or len(result1["embeddings"]) == 0:
        print("  ✗ DB1 has no embeddings!")
        return False
    
    if result2["embeddings"] is None or len(result2["embeddings"]) == 0:
        print("  ✗ DB2 has no embeddings!")
        return False
    
    print(f"  DB1: {len(result1['embeddings'])} embeddings")
    print(f"  DB2: {len(result2['embeddings'])} embeddings")
    
    # Compare first embedding
    emb1 = np.array(result1["embeddings"][0])
    emb2 = np.array(result2["embeddings"][0])
    
    print(f"\n  First embedding comparison:")
    print(f"    DB1 shape: {emb1.shape}")
    print(f"    DB2 shape: {emb2.shape}")
    print(f"    DB1 first 5 values: {emb1[:5]}")
    print(f"    DB2 first 5 values: {emb2[:5]}")
    
    # Check if identical
    if np.allclose(emb1, emb2):
        print(f"  ✗ EMBEDDINGS ARE IDENTICAL! Something is wrong.")
        return False
    else:
        # Calculate similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"  ✓ EMBEDDINGS ARE DIFFERENT")
        print(f"    Cosine similarity: {similarity:.4f}")
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_embeddings.py <db1_path> <db2_path>")
        sys.exit(1)
    
    db1_path = sys.argv[1]
    db2_path = sys.argv[2]
    
    print(f"DB1: {db1_path}")
    print(f"DB2: {db2_path}")
    print("="*60)
    
    # Create clients ONCE with consistent settings
    print("Connecting to databases...")
    settings = Settings(allow_reset=True)
    client1 = chromadb.PersistentClient(path=db1_path, settings=settings)
    client2 = chromadb.PersistentClient(path=db2_path, settings=settings)
    
    # Get collections from DB1
    collections = get_all_collections(client1)
    print(f"\nFound {len(collections)} collections in DB1")
    
    # Compare each collection
    results = {}
    for collection in collections:
        result = compare_embeddings(client1, client2, collection)
        results[collection] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    
    for collection, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {collection}")
    
    print(f"\n{success_count}/{total_count} collections passed")
    
    # Exit with appropriate code
    sys.exit(0 if success_count == total_count else 1)