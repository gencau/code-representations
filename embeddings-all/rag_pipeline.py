from dotenv import load_dotenv
import os
import csv
from datetime import datetime
from pathlib import Path

from utils.dataset_utils import DatasetUtils as data, Record
from database.similarity_search_full_file import SimilaritySearch
from database.record_processor import RecordProcessor


def main(datasets: list[str],
         dataset_paths: list[Path],
         repo_path: Path,
         stratified_sampling: bool = False,
         no_add: bool = False):
    # Load environment variables from .env file
    load_dotenv()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile_name = f"results/results_{timestamp}.csv"

    out_path = Path("results")
    out_path.mkdir(parents=True, exist_ok=True)

    total_errors = 0

    languages = ['py', 'java', 'kt']
    for ds_name, ds_path in zip(datasets, dataset_paths):     

        for language in languages:
            print(f"Processing {language} records...")
            if ds_name == "swe_verified" and language != "py":
                continue          # that dataset is Python-only

            print(f"Processing language: {language} for {ds_name}")

            if ds_name == "swe_verified":
                database_name = "/Volumes/T9/rq1/filepaths/swe/bgelarge"
            else:
                #database_name = "/Volumes/T9/summaries/prompt-variations/lca/30_tokens/file-level/all/queries_5_test_codexembed"
                database_name = "/Volumes/T9/all_results/rq1-embeddings/trino_times/queries/bgelarge"
                #database_name = "/Volumes/T9/rq1/chroma_db_embeddings_512_token_lca"

            records = data.get_records(ds_name, ds_path, repo_path, language, stratified_sampling=stratified_sampling)
            print(f"Retrieved {len(records)} records for language: {language} from dataset: {ds_name}")

            total_errors += process_dataset(records, database_name, language, no_add, outfile_name) 


    print(f"Total errors across all datasets and languages: {total_errors}")

def process_dataset(records : list[Record], database_name, language, no_add, outfile_name):
    errors = 0
    for record in records: 
        if int(record.id) != 8247:
            continue
        print(f"Processing record ID: {record.id} from repo {record.repo_name}")
        print(f"Bug description: {record.bug_description}")
        print(f"Real diff: {record.diff}")
        process_rec = True

        rag = SimilaritySearch(path=database_name, collection_name=record.base_sha)

        if (rag.collectionExists() == True or no_add == True):
            # collection already exists
            print(f"Collection {record.base_sha} already exists or not adding.")
            process_rec = False

        rag.createOrSetCollection()

        processor = RecordProcessor(representation_type='summary', chunking_granularity='file')
        indexing_time = 0
        summary_processing_time = 0

        if process_rec:
            # Process record and extract data and embeddings
            start_time = datetime.now()
            record_data, file_embeddings = \
                processor.process_record(record.location, language, batch_size=32)
            end_time = datetime.now()
            indexing_time = (end_time - start_time).total_seconds()
            summary_processing_time = record_data.get("summary_processing_time", 0)
            errors += record_data.get("errors", 0)

            rag.add_to_chroma_in_batches(record_data['file_summaries'], file_embeddings, record_data['file_metadata'], record_data['file_ids'])
            print(f"Indexing completed in {indexing_time:.2f} seconds.")
            print("File records stored in database.")

        results, search_time = search_database(record, rag, processor)
        write_results(record, results, outfile_name, indexing_time, search_time, summary_processing_time)
    return errors

def search_database(record, rag, processor):
    results = []
    elapsed_time = 0
    #  There is no point in searching if the embeddings are not in the db
    if (rag.collectionExists() == True):
        print("Searching for similar chunks...")
        # Process bug description and search for similar chunks
        start_time = datetime.now()
        bug_embeddings = processor.embed_bug_description(record.bug_description)
        results = rag.search_similar_chunks(bug_embeddings, top_k=5)
        end_time = datetime.now()
        print(f"Found {len(results)} similar chunks.")

        elapsed_time = (end_time - start_time).total_seconds()
        print(f"Search completed in {elapsed_time:.2f} seconds.")

    else:
        print(f"Collection {record.base_sha} does not exist, skipping search.")  

    return results, elapsed_time

def write_results(record, results, outfile_name, indexing_time, search_time, summary_processing_time):
    documents = []
    topkfiles = []
    distances = []
    
    if len(results) > 0:
        line = ({'index':record.id, "metadata":results[0][1]})
        file_exists = os.path.isfile(outfile_name)

        for i, (doc, metadata, distance) in enumerate(results, start=1):
            documents.append(doc)
            topkfiles.append(metadata['filename'])
            distances.append(distance)

        with open(outfile_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['index', 'repo_name', 'db_key', 'bug_description', 'documents', 'metadata', 'distances', \
                                    'topk_files', 'expected_files', 'indexing_time', 'search_time', 'summary_processing_time'])

            writer.writerow([line['index'], record.repo_name, record.base_sha, record.bug_description, documents, line['metadata'], \
                            distances, topkfiles, record.changed_files, indexing_time, search_time, summary_processing_time])


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",
                        default="lca",
                        help="comma-separated: lca,swe_verified …")
    parser.add_argument("--dataset_paths",
                        default=".",          # same position-wise order
                        help="comma-separated absolute paths to parquet file")
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--stratified_sampling", type=bool, default=False, help="If set, use stratified sampling across languages (if applicable) to select records from the dataset.")
    parser.add_argument("--no_add", type=bool, default=False,
                        help="If set, do not add new collections to the database. Use when you just want to run similarity search.")

    args = parser.parse_args()

    ds_list = args.datasets.split(",")
    path_list = [Path(p).expanduser() for p in args.dataset_paths.split(",")]

    # allow the user to specify just one path that will be reused
    if len(path_list) == 1 and len(ds_list) > 1:
        path_list *= len(ds_list)

    main(ds_list, path_list, args.repo_path, args.stratified_sampling, args.no_add)