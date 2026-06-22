import shutil
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

class Storage:
    def __init__(self, path, collection_name):
        self.chroma_client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.collection = None

    def create_collection(self):
        self.collection = self.chroma_client.create_collection(name=self.collection_name,
                                                               metadata={"hnsw:space":"cosine"}, 
                                                               get_or_create=True
                                                               )
    def set_collection(self):
        self.collection = self.chroma_client.get_collection(self.collection_name)
        
    def collection_exists(self) -> bool:
        try:
            # Attempt to get the collection
            collection = self.chroma_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists.")
            return True
        except:
            return False

    def get_chroma_client(self):
        return self.chroma_client
    
    def add(self, texts, embeddings, metadatas, ids):
        self.collection.add(documents=texts,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids)

    # Returns the top k most similar documents to the query_text
    def query(self, query_embeddings, top_k=5):
        return self.collection.query(query_texts=query_embeddings, n_results=top_k, 
                                     include=["documents", "metadatas", "distances"]
                                    )
    
    def similarity_search_with_score(self, query_embedding, k=5, filter=None):
        # Prepare the query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k, 
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter:
            # ChromaDB expects filters to be under the 'where' key
            query_params["where"] = filter
        
        # Perform the query
        results = self.collection.query(**query_params)
        
        # Extract documents, metadatas, and distances
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Combine the results into a list of tuples
        combined_results = list(zip(documents, metadatas, distances))
        
        return combined_results
    
    def cleanup(self):
        self.chroma_client.reset()
        shutil.rmtree("chroma_db")

