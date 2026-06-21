AGENT_PROMPT_TEMPLATE = """
    Issue: {}
    You are given bug issue description.
    Select subset of 1-5 files which SHOULD be fixed according to issue.
    Start with repo exploration by listing files in directory (start with root). 
    Read bug-related files to make sure they contain bugs.
    Provide output in JSON format with one field 'files' with list of file names which SHOULD be fixed.
    Provide ONLY json without any additional comments.
"""

# To be used to calculate the number of tokens in this part (fixed between iterations)
AGENT_SUMMARY_PROMPT_TEMPLATE = """
You will be given a list of files along with their summary. Select the files that SHOULD be fixed based on the issue description and the file summaries.

Output MUST be a valid JSON object with a single field 'files', containing either no files or a list of files that need fixing. Do NOT include any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

Example output with 2 files:
{{ "files": ["dir1/file1.py", "dir2/file2.py"] }}

Example output with no files:
{{ "files": []}}

<ISSUE_DESCRIPTION>
{}
</ISSUE_DESCRIPTION>

<FILES>
{}
</FILES>
"""

AGENT_CONTEXT_PROMPT_TEMPLATE = """
You will be given a list of files. Select the files that SHOULD be fixed based on the issue description.

Output MUST be a valid JSON object with a single field 'files', containing either no files or a list of files that need fixing. Do NOT include any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

Example output with 2 files:
{{ "files": ["dir1/file1.py", "dir2/file2.py"] }}

Example output with no files:
{{ "files": []}}

<ISSUE_DESCRIPTION>
{}
</ISSUE_DESCRIPTION>

<FILES>
{}
</FILES>
"""

AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are a software developer specialized in finding files that contain bugs given a bug description.
"""

AGENT_RERANK_PROMPT_TEMPLATE = """
You are given a list of files selected as potential candidates to fix the provided issue.
Your task is to re-rank these files, from most important to least important for addressing the issue. 

**Do not remove any files from the list; simply reorder them based on their relevance.**

Output MUST be a valid JSON object with a single field 'files', containing the list of reranked files. Do NOT include any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

<ISSUE>
{issue}
</ISSUE>

<FILES>
{files}
</FILES>
"""

AGENT_RERANK_SUMMARIES_PROMPT_TEMPLATE = """
You are given a list of files previously selected as potential candidates to fix the provided issue, along with a summary of each file.
Your task is to re-rank these files, from most important to least important for addressing the issue.
**Do not remove any files from the list; simply reorder them based on their relevance.**

Output MUST be a valid JSON object with a single field 'files', containing the list of reranked files. The "files" field must be an array of file paths and names (exactly as given in the input) that require fixes.
Do NOT include the summaries or any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

<ISSUE>
{issue}
</ISSUE>

<FILES>
{files}
</FILES>
"""

AGENT_CODE_PROMPT_TEMPLATE_2 = """
You will receive two pieces of input:
1. A bug description.
2. A list of files. Each file includes its relative path, file name, and content.
Your task:
- Read the bug description.
- Check the content of each file.
- Decide which file or files need to be fixed because they are related to the bug.
Output instructions:
- Return ONLY a JSON object.
- This JSON object must have one key: "files".
- The value for "files" must be an array of file paths and names (exactly as given in the input) that require fixes.
- If no file needs fixing, use an empty array.
- Do not include any extra text, comments, or explanations.
Example of valid output if files need fixing:
{{"files": ["src/example/file1.py", "tests/test_file2.py"]}}
Example of valid output if no files need fixing:
{{"files": []}}

<ISSUE_DESCRIPTION>
{issue}
</ISSUE_DESCRIPTION>

<FILES>
{files}
</FILES>
"""

AGENT_CODE_RERANK_PROMPT_TEMPLATE = """
You are given a list of files previously selected as potential candidates to fix the provided issue, along with their content.
Your task is to re-rank these files given their content, from most important to least important for addressing the issue given the issue description.
**Do not remove any files from the list; simply reorder them based on their relevance.**

Output MUST be a valid JSON object with a single field 'files', containing the list of reranked files. The "files" field must be an array of file paths and names (exactly as given in the input) that require fixes.
Do NOT include the file contents or any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

<ISSUE>
{issue}
</ISSUE>

<FILES>
{files}
</FILES>
"""