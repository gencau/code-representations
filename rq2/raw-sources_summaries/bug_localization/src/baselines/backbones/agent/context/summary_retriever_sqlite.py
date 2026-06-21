import sqlite3

class SummaryRetrieverSQLite:
    """
    SummaryRetriever replaced to use direct SQLite queries instead of Chroma API.
    This was necessary because of a corruption in the chroma indexes.
    With a healthy database, use SummaryRetriever.
    """
    def __init__(self, database_location: str):
        """
        Initialize with the path to your SQLite database file.
        """
        self._database_path = database_location

    def findSummary(self, collection_name: str, filename: str) -> dict:
        """
        Retrieve documents and metadata for the given collection and filename.

        Returns a dict with keys:
          - 'documents': list of document text strings
          - 'metadatas': list of metadata dicts for each document
        """
        results = {'documents': [], 'metadatas': []}

        conn = sqlite3.connect(self._database_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Step 1: find matching document IDs
        query_ids = '''
        SELECT e.id
          FROM embedding_metadata AS em
          JOIN embeddings            AS e ON em.id        = e.id
          JOIN segments              AS s ON e.segment_id = s.id
          JOIN collections           AS c ON s.collection = c.id
         WHERE c.name         = ?
           AND em.key          = 'filename'
           AND em.string_value = ?
        '''
        cur.execute(query_ids, (collection_name, filename))
        id_rows = cur.fetchall()
        if not id_rows:
            conn.close()
            print(f"Could not find matching documents for {filename}")
            return results

        for row in id_rows:
            doc_id = row['id']

            # Step 2: fetch document text from the FTS5 table
            query_text = 'SELECT c0 AS document_text FROM embedding_fulltext_search_content WHERE rowid = ?'
            cur.execute(query_text, (doc_id,))
            text_row = cur.fetchone()
            doc_text = text_row['document_text'] if text_row else ''

            # Step 3: fetch all metadata for this document
            query_meta = 'SELECT key, string_value FROM embedding_metadata WHERE id = ?'
            cur.execute(query_meta, (doc_id,))
            meta_dict = {r['key']: r['string_value'] for r in cur.fetchall()}

            results['documents'].append(doc_text)
            results['metadatas'].append(meta_dict)

        conn.close()
        return results
