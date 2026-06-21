import llm.prompts as prompts
import json


def generateFileSummary(text : str, llm_client) -> str:
    prompt = prompts.generateFileSummary2(text)
    docstring = ""

    try:
        docstring = llm_client.formatAndSend(prompt)
    except RuntimeError as e:
        print("Request failed after retries:", str(e))
    
    return docstring

def generateClassSummary(text: str, llm_client) -> str:
    prompt = prompts.generateClassSummary(text)
    docstring = ""

    try:
        docstring = llm_client.formatAndSend(prompt)
    except RuntimeError as e:
        print("Request failed after retries:", str(e))
    
    return docstring

def generateMethodSummary(text: str, llm_client) -> str:
    prompt = prompts.generateMethodSummary(text)
    docstring = ""

    try:
        docstring = llm_client.formatAndSend(prompt)
    except RuntimeError as e:
        print("Request failed after retries:", str(e))
    
    return docstring

def generateRetrievalQueries(text: str, llm_client, num_queries: int = 5, num_tokens: int = 30) -> list[str]:
    prompt = prompts.generateQueriesPrompt(text, num_queries, num_tokens)

    print(f"Generated prompt for retrieval queries:\n{prompt}\n")

    queries = []

    try:
        response = llm_client.formatAndSend(prompt)
        # Parse the JSON response to extract the queries list
        print(f"Raw response for retrieval queries:\n{response}\n")
        queries = json.loads(response)
    except (RuntimeError, json.JSONDecodeError) as e:
        print("Request failed or response parsing error:", str(e))
    
    return queries

def generateBugReports(text: str, llm_client, num_reports: int = 5,  num_tokens=30) -> list[str]:
    prompt = prompts.generateQueriesSimple3(text, num_queries=num_reports, num_tokens=num_tokens)

    print(f"Generated prompt for bug reports:\n{prompt}\n")

    bug_reports = []

    try:
        response = llm_client.formatAndSend(prompt)
        # Parse the JSON response to extract the bug reports list
        print(f"Raw response for bug reports:\n{response}\n")
        bug_reports = json.loads(response)
    except (RuntimeError, json.JSONDecodeError) as e:
        print("Request failed or response parsing error:", str(e))
    
    return bug_reports