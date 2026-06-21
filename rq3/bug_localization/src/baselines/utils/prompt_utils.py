import json
import re
from typing import List, Dict, Any, Optional

from src.baselines.backbones.chat.prompts.chat_base_prompt import ChatBasePrompt
from src.utils.tokenization_utils import TokenizationUtils


def check_match_context_size(tokenization_utils: TokenizationUtils,
                             prompt: ChatBasePrompt,
                             issue_description: str,
                             project_content: Dict[str, str],
                             is_chat: bool):
    if is_chat:
        messages = prompt.chat(issue_description, project_content)
        return tokenization_utils.messages_match_context_size(messages)

    text = prompt.complete(issue_description, project_content)
    return tokenization_utils.text_match_context_size(text)


def batch_project_context(model: str,
                          prompt: ChatBasePrompt,
                          issue_description: str,
                          project_content: Dict[str, str],
                          is_chat: bool) -> List[Dict[str, str]]:
    tokenization_utils = TokenizationUtils(model)
    file_paths = list(project_content.keys())

    has_big_message = True
    n = len(file_paths)
    step = len(file_paths)

    while has_big_message:
        has_big_message = False
        for i in range(0, n, step):
            project_content_subset = {f: c for f, c in project_content.items() if f in file_paths[i:i + step]}
            if not check_match_context_size(tokenization_utils, prompt, issue_description, project_content_subset,
                                            is_chat):
                has_big_message = True
                step //= 2
                break

    batched_project_content = [
        {f: c for f, c in project_content.items() if f in file_paths[i:i + step]} for i in range(0, n, step)
    ]
    assert len(file_paths) == sum(len(b) for b in batched_project_content)

    return batched_project_content

def deduplicate_files(files):
    """
        Removes duplicates and invalid formats (dict) inserted within JSON objects.
    """
    unique = []
    seen = set()
    
    for item in files:
        # For dictionaries, convert to a tuple of sorted (key, value) pairs.
        if isinstance(item, dict):
            #skip it
            continue
        else:
            # Assuming item is already hashable (like a string)
            key = item
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def parse_json_response(response: str) -> list:
    files = []
    patterns = [
        r"```json\s*(\{.*?\})\s*```",   # JSON inside markdown code blocks
        r'</think>\s*(\{.*?\})',         # JSON after a </think> marker
        r"(\{.*?\})"                    # Fallback: any JSON-like object
    ]
    
    # Extract and process JSON objects
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for json_str in matches:
            try:
                data = json.loads(json_str)

                files_field = data.get("files")
                if isinstance(files_field, list):        # normal, expected case
                    files.extend(files_field)
                elif isinstance(files_field, str):       # single filename
                    files.append(files_field)
                elif files_field is None:                # nothing returned
                    print("#### 'files' is null for match:", json_str)
                else:                                    # some other unexpected type
                    print("#### Unexpected 'files' type:", type(files_field))
            except json.JSONDecodeError:
                print("#### JSON decoding error for match:", json_str)

    # Remove markdown code blocks before inline extraction to avoid double matching.
    cleaned_result = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
    # Also remove <think> tags so that we don't match files mentioned there
    cleaned_result = re.sub(r"<think>.*?</think>", "", cleaned_result, flags=re.DOTALL)
    if len(files) == 0:
        inline_matches = re.findall(r'`([^`]+)`', cleaned_result)
        # Filter inline matches: accept only candidates that appear to contain files (i.e. contain '.')
        valid_inline_matches = [m for m in inline_matches if '.' in m]
        files.extend(valid_inline_matches)

    # Get rid of any duplicates
    files = deduplicate_files(files)

    return files


def parse_list_files_completion(raw_completion: str,) -> List[str]:
    return parse_json_response(raw_completion)
