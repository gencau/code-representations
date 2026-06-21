FILE_LIST_PROMPT_TEMPLATE = """
You will be given a list of files. Select 1-5 files that SHOULD be fixed based on the issue description. 

Output MUST be a valid JSON object with a single field 'files', containing either a single file or a list of up to 5 file names including their relative path. Do NOT include any additional text, comments, or explanations.
Each file must be specified only once in the list. Double check that the file names you include are either part of the list of files provided, or mentioned in the bug description and part of the project.

Example output:
{{ "files": ["/dir1/file1.py", "/dir2/file2.py"] }}

Provide ONLY the JSON output.

Bug description: {}
List of files: {}
"""

FILE_LIST_PROMPT_TEMPLATE_2 = """
Your task is to identify between 1 and 5 files that likely need to be fixed to resolve the bug. Follow these steps:
1. Review the list of files and the bug description.
2. Select the most relevant files (choose at least 1 and at most 5).
3. Double check that the file names you include are part of the list of files provided.
4. Don't include duplicate file names in your answer.
5. Return your answer as valid JSON in the following format:
   {{"files": ["/dir1/file1", "/dir2/file2", ...]}}

Important: Output only the JSON object. Do not include any extra text or commentary.

Bug issue description: {}
List of file names: {}
"""

