import chromadb
from chromadb.config import Settings

class SummaryRetriever():

    def __init__(self, database_location: str):
        self._database_location = database_location
        self.chroma_client = chromadb.PersistentClient(path=self._database_location, settings=Settings(allow_reset=True))

    def find_summary(self, collection: str, filename: str) -> dict:
        try:
            # Attempt to get the collection
            collection = self.chroma_client.get_collection(collection)
            print(f"Collection '{collection}' exists.")
        except Exception as e:
            print(f"Error retrieving collection: {e}")
            return {}

        results = collection.get(where={"filename": filename})
        print(f"Found {len(results['ids'])} in collection")
        return results