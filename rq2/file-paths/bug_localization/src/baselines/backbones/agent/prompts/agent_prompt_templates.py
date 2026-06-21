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
AGENT_ISSUE_PROMPT_TEMPLATE = """
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
"""

AGENT_CONTEXT_PROMPT_TEMPLATE = """
<FILES>
{}
</FILES>
"""

AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are a software developer specialized in finding files that contain bugs given a bug description.
"""

AGENT_RERANK_PROMPT_TEMPLATE = """
You are given a list of files selected as potential candidates to fix the provided issue.
Re-rank the files, in order of most important to modify. You may remove files from the list if you think they should not be included.

Return the re-ranked list of files in JSON format with a single field 'files'.

<ISSUE>
{}
</ISSUE>

<FILES>
{}
</FILES>
"""

AGENT_RERANK_PROMPT_TEMPLATE_2 = """
You are given a list of files previously selected as potential candidates to fix the provided issue.
Your task is to re-rank these files, from most important to least important for addressing the issue.
**Do not remove any files from the list; simply reorder them based on their relevance.**

Output MUST be a valid JSON object with a single field 'files', containing the list of reranked files. Do NOT include any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

Provide ONLY the JSON output.

<ISSUE>
{}
</ISSUE>

<FILES>
{}
</FILES>
"""