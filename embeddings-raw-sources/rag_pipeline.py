from dotenv import load_dotenv
import os
import argparse
import csv
from pathlib import Path
from datetime import datetime

from utils.dataset_utils import DatasetUtils as data
from database.similarity_search import SimilaritySearch
from utils.embed_utils import build_python_dependency_graph, build_java_kt_dependency_graph

def main(datasets: list[str],
         dataset_paths: list[Path],
         repo_path: Path,
         percent: int = 0,
         include_dependencies: bool = False,
         embed_location: str = "/tmp",
         stratified_sampling: bool = False) -> None:

    # Load environment variables from .env file
    load_dotenv()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    languages = ['py', 'java', 'kt']

    for ds_name, ds_path in zip(datasets, dataset_paths): 
        outfile_name = f"results/{ds_name}/results_{timestamp}.csv"  
        # Create results directory if it doesn't exist
        outfile = Path(outfile_name)
        outfile.parent.mkdir(parents=True, exist_ok=True)     
        
        for language in languages:
            if ds_name == "swe_verified" and language != "py":
                continue          # that dataset is Python-only

            print(f"Processing language: {language} for {ds_name}")

            database_name = embed_location

            records = data.get_records(ds_name, ds_path, repo_path, language, 0, stratified_sampling)

            for record in records: 

                print(f"Processing record ID: {record.id} from repo {record.repo_name}")
                print(f"Bug description: {record.bug_description}")
                print(f"Real diff: {record.diff}")
                processRecord = True

                rag = SimilaritySearch(path=database_name, collection_name=record.base_sha)

                if (rag.collectionExists() == True):
                    # collection already exists
                    print(f"Collection {record.base_sha} already exists.")
                    processRecord = False

                rag.createOrSetCollection()
                
                k=5
                if percent > 0:
                    k = max(1, int((percent / 100.0) * (record.repo_files_without_test_count)))
                    print(f"Top k: {k}")
                
                if processRecord:
                    start_time = datetime.now()
                    rag.processRecord(record.location, language=language)
                    end_time = datetime.now()
                    print(f"Time taken to process record {record.id}: {end_time - start_time}")

                start_time = datetime.now()
                results = rag.search_similar_chunks(record.bug_description, k=k, retrieve_max=record.repo_files_without_test_count)
                end_time = datetime.now()
                print(f"Time taken to search similar chunks for record {record.id}: {end_time - start_time}")

                documents = []
                topkfiles = []
                distances = []
                filenames = []
                seen = set()

                if len(results) > 0:
                    line = ({'index':record.id})
                    file_exists = os.path.isfile(outfile_name)

                    for i, (doc, metadata, distance) in enumerate(results, start=1):
                        if doc == None or metadata == None:
                            continue

                        # Deduplicate the results, since we're only interested in different files found at this point
                        key = (metadata.get('directory'), metadata.get('filename'))
                        if key not in seen:
                            filenames.append(metadata)
                            documents.append(doc)
                            topkfiles.append(metadata['filename'])
                            distances.append(distance)
                            seen.add(key)

                    callers = []
                    callees = []
                    all_files = []
                    if include_dependencies:
                        fullpath = Path(record.location).resolve()
                        print(f"Record location: {fullpath}")
                        print(f"Number of files in list: {len(filenames)}")
                        print(f"Files in list: {topkfiles}")

                        if language == "py":
                            graph = build_python_dependency_graph(fullpath)
                        else:
                            graph = build_java_kt_dependency_graph(fullpath)

                        callers = []
                        callees = []
                        all_files = []
                        for file in topkfiles:
                            callers = graph.in_edges(file)
                            callees = graph.out_edges(file)

                        print(f"Callers before processing tupes: {callers}")
                        print(f"Callees before processing tuples: {callees}")

                        # Make sure we don't get a list of tuples
                        if len(callers) > 0:
                            # Flatten the list
                            flattened_callers = [item for tup in callers for item in tup]
                            callers = list(dict.fromkeys(flattened_callers))
                        if len(callees) > 0:
                            flattened_callees = [item for tup in callees for item in tup]
                            callees = list(dict.fromkeys(flattened_callees))
                        callers_unique = set(callers)
                        callees_unique = set(callees)

                        all_files.extend(topkfiles)
                        all_files.extend(callers_unique)
                        all_files.extend(callees_unique)
                        print(f"Got callers: {callers_unique} and callees: {callees_unique} for {record.repo_name}")

                        # Get retrieval results with a K equivalent to the number of unique files when including dependencies
                        new_k = len(set(all_files))
                        print(f"New k: {new_k}")
                        baseline_results = rag.search_similar_chunks(record.bug_description, k=new_k, max_retrieve=record.repo_files_without_test_count)
                        baseline_documents = []
                        baseline_topkfiles = []
                        baseline_distances = []
                        seen = set()
                        for i, (doc, metadata, distance) in enumerate(baseline_results, start=1):
                            if doc == None or metadata == None:
                                continue

                            # Deduplicate the results, since we're only interested in different files found at this point
                            key = (metadata.get('directory'), metadata.get('filename'))
                            if key not in seen:
                                baseline_documents.append(doc)
                                baseline_topkfiles.append(metadata['filename'])
                                baseline_distances.append(distance)
                                seen.add(key)

                    with open(outfile_name, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(['index', 'repo_name', 'db_key', 'documents', 'filename', 'distance', \
                                            'topk_files', 'expected_files', 'all_files'])#, 'baseline_all_files', 'baseline_distances'])

                        writer.writerow([line['index'], record.repo_name, record.base_sha, documents, filenames, \
                                        distances, topkfiles, record.changed_files, all_files])#, baseline_topkfiles, baseline_distances])
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",
                        default="lca",
                        help="comma-separated: lca,swe_verified …")
    parser.add_argument("--dataset_paths",
                        default=".",          # same position-wise order
                        help="comma-separated absolute paths")
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--percent", type=int, default=0)
    parser.add_argument("--include_dependencies", type=bool, default=False)
    parser.add_argument("--embed_location", type=str, required=True)
    parser.add_argument("--stratified_sampling", type=bool, default=False)
    args = parser.parse_args()

    ds_list = args.datasets.split(",")
    path_list = [Path(p).expanduser() for p in args.dataset_paths.split(",")]

    # allow the user to specify just one path that will be reused
    if len(path_list) == 1 and len(ds_list) > 1:
        path_list *= len(ds_list)

    main(ds_list, path_list, args.repo_path, args.percent, args.include_dependencies, 
        args.embed_location, args.stratified_sampling)