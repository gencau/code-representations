
AGENT_BUG_REPORT_PROMPT_TEMPLATE = """
You will be given a list of files along with summaries of potential bug reports that might be written for each file (up to 5 summaries per file). Select the files that SHOULD be fixed based on the issue description and the file summaries.

Output MUST be a valid JSON object with a single field 'files', containing either no files or a list of files that need fixing. Do NOT include any additional text, comments, or explanations.
Mention each file only once. Double check that the paths and file names you include are exactly as provided in the list of files.

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

Provide ONLY the JSON output.
"""

AGENT_BUG_REPORT_RANK_PROMPT_TEMPLATE = """
You are given a list of files previously selected as potential candidates to fix the provided issue, along with their content.
Each file is associated with a list of potential bug report summaries that might be written for that file.
Your task is to re-rank these files given their bug report summaries, from most important to least important for addressing the issue given the issue description.
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

AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are a software developer specialized in finding files that contain bugs given a bug description.
"""
