

def getDocstringSummaryPrompt(docstring) -> str:
    prompt = f"Summarize the following docstring in a concise single sentence. Make sure to include all important information. Provide only the docstring, with no explanations or tags. Docstring to summarize: {docstring}."
    return prompt

def getCodeSnippetPrompt(text) -> str:
    prompt = f'''If there is a code snippet in the following text, return it within triple backticks with no explanation. 
                 If there is no code snippet, return "None" with no other explanation.
                 Example 1: 
                 I believe there is a bug in this code:
                 def add(a, b):
                     return a + b

                 In this case, return ```def add(a, b): return a + b```

                 Example 2:
                 I believe there is a bug in this code.
                 
                 In this case, return ```None```
                 
                 Here is the text to analyze:{text}'''
    return prompt

