from langchain_text_splitters import RecursiveCharacterTextSplitter
from database.storage import Storage

class SimilaritySearch:
    def __init__(self, path, collection_name):
        # instance of the database
        self.storage_instance = Storage(path, collection_name=collection_name)
        self.collection_name = collection_name

    def createOrSetCollection(self):
        try:
            self.storage_instance.create_collection()
        except Exception as e:
            print(f"Error creating collection: {e}")
            self.storage_instance.set_collection()

    def collectionExists(self) -> bool:
        return self.storage_instance.collection_exists()

    def add_to_chroma_in_batches(
        self,
        documents,
        embeddings,
        metadatas,
        ids,
        chroma_batch_size=1000
    ):
        for i in range(0, len(documents), chroma_batch_size):
            self.storage_instance.add(
                texts=documents[i:i + chroma_batch_size],
                embeddings=embeddings[i:i + chroma_batch_size],
                metadatas=metadatas[i:i + chroma_batch_size],
                ids=ids[i:i + chroma_batch_size],
            )

    def store_to_database(self, file_summaries, files_embeddings, files_metadata, ids):
        if len(file_summaries) > 0:
            self.storage_instance.add(
                texts=file_summaries,
                embeddings=files_embeddings,
                metadatas=files_metadata,
                ids=ids
            )
        else:
            print("-------- No chunks generated, maybe a parse error?")

    def search_similar_chunks(self, bug_description_embeddings, top_k=100):
        # Perform a similarity search in the database
        results = []
        for i, text_embedding in enumerate(bug_description_embeddings):
            text_results = self.storage_instance.similarity_search_with_score(text_embedding, k=top_k)
            results.extend(text_results)

        return results
    
    def cleanup(self):
        # For now, cleanup when exiting
        self.storage_instance.cleanup()