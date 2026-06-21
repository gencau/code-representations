

def getDocstringSummaryPrompt(docstring) -> str:
    prompt = f"Summarize the following docstring in a concise single sentence. Make sure to include all important information. Return only the docstring within triple backticks, with no explanations or tags. Docstring to summarize: {docstring}."
    return prompt

def generateDoctringPrompt(text) -> str:
    prompt = f"Generate a concise docstring summarized in one line that explains the provided code snippet. Make sure to include all important information. Return only the docstring with no explanation, tags or mention of the programming language. Do not put the programming language name anywhere. Code snippet: {text}."
    return prompt

def generateFileSummary(text) -> str:
    prompt = f'''
                <file-content>
                    {text}
                </file-content>
    
                Give a concise summary of this chunk of code to situate it within the project and the context of the file.
                The purpose is to improve search retrieval of this chunk.
                Include relevant methods, classes and functionality.
                Output only the summary text without any additional explanation, formatting tags, or commentary.
    '''
    return prompt

def generateFileSummary2(text) -> str:
    prompt = f'''
                <file-content>
                    {text}
                </file-content>

                Give a concise summary of this chunk of code to situate it within the project and the context of the file.
                The purpose is to improve search retrieval of this chunk.
                Include relevant methods, classes and functionality.
                Focus on the file's role and its main responsibilities.
                Output only the summary text without any additional explanation, formatting tags, or commentary.
            '''
    return prompt

def generateFileSummary3(text) -> str:
    prompt = f'''
            <file-content>
                {text}
            </file-content>

            Generate a comprehensive summary of this source file for retrieval purposes. 
            
            The purpose is to improve search retrieval of this chunk.
            Include:
            1. **Primary purpose**: What problem does this file solve? What is its role in the codebase?

            2. **Key components**: List main classes, functions, and their responsibilities. Include:
            - Class names and their inheritance relationships
            - Public API methods and their parameters
            - Important data structures or state management

            3. **Technical signals**: Explicitly mention:
            - Dependencies and imports (external libraries, internal modules)
            - Error types thrown or handled (exceptions, error codes)
            - Configuration or environment variables used
            - File I/O operations, database queries, or network calls
            - Design patterns implemented (singleton, factory, observer, etc.)

            4. **Interactions**: Describe how this file relates to others (what it calls, what calls it)

            5. **Notable implementation details**: Algorithms, edge cases handled, performance considerations, or non-obvious behaviors

            Use technical vocabulary and specific identifiers from the code. 
            Output only the summary text without any additional explanation, formatting tags, or commentary.
            '''
    return prompt

def generateFileSummary4(text) -> str:
    prompt = f'''
     <file-content>
        {text}
    </file-content>
    
    In 1-2 sentences, provide a condensed summary of this file.
    The purpose is to improve search retrieval of this chunk.
    Include relevant classes, methods, and functionality. 
    Include key dependencies.
    Preserve all function/class names verbatim.
    Output only the summary text without any additional explanation, formatting tags, or commentary.
    '''
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

def generateClassSummary(text) -> str:
    prompt = f'''
                <class>
                    {text}
                </class>
    
                Provide a concise summary that captures the core functionality of the class. 
                Focus on the class's role and its main responsibilities, and output only the summary text without any additional explanation, formatting tags, or commentary.
    '''
    return prompt

def generateMethodSummary(text) -> str: 
    prompt = f'''
                <method>
                    {text}
                </method>
    
                Provide a concise summary that captures the core functionality of the method. 
                Describe the parameters, return values, and any significant code patterns. 
                Give a sample usage and output.
                Focus on the method's role and its main responsibilities, and output only the summary text without any additional explanation, formatting tags, or commentary.
    '''
    return prompt

"""
Results with this prompt are better mostly because of a different temperature used (probably): 0.6 vs 0.0
With 0.6 and this prompt on 10 python repos, we get MAP 0.258 and Hit@5 0.3
"""
def generateBugReport(snippet, num_queries, num_tokens) -> str:
    prompt = f"""
        Given the following code file:
        <code-snippet>
        {snippet}
        </code-snippet>

        Task: Generate {num_queries} DISTINCT retrieval expansions that resemble minimal bug-report fragments a developer might write when THIS file is responsible.

        Goal: Produce anchor-rich fragments that maximize lexical overlap (BM25) with real bug reports while remaining semantically plausible.

        Hard constraints:
        - Do NOT write a full bug report. No narrative, no steps, no expected/actual sections.
        - Each expansion must be <= {num_tokens} tokens.
        - Each expansion must include at least TWO anchors copied VERBATIM from the snippet.
        - Avoid generic phrases ("bug", "issue", "doesn't work") unless paired with concrete anchors and a symptom.
        - Ensure diversity across different failure modes.
        - Do not invent inputs or behaviors not implied by the snippet. If you mention a specific exception, it must be consistent with the code.
        
        Quality check: If you cannot satisfy the anchor + symptom constraints for an expansion, output fewer than {num_queries} items.

        Return ONLY a JSON array of strings:
        ["exp1", "exp2", ..., "exp{num_queries}"]
    """

    return prompt

# Let's give this one a try first
# This one was used for the 32 repos stratified per language.
def generateQueriesPrompt(snippet: str, num_queries: int, num_tokens: int) -> str:
    prompt = f'''
        Given the following code file: 
        <code-snippet>
        {snippet}
        </code-snippet>

        You are given a single source code file. Generate {num_queries} DISTINCT retrieval expansions that resemble the *minimal text fragments* a developer might write in a bug report when this file is responsible.
        Each expansion should contains unique code elements in this snippet that could be mentioned in a bug report.
        
        Goal: maximize match under BM25 and dense embeddings between real bug reports and these expansions.

        Constraints:
        - Do NOT write a full bug report. Do NOT include narrative, steps to reproduce, or expected/actual sections.
        - Each expansion must be short: <= {num_tokens} tokens (hard limit).
        - Each expansion must include at least TWO concrete anchors from the file:
        - function/method names, class names, key identifiers, exception names, constants, config keys, API endpoints, or domain terms present in code.
        - Prefer symptom tokens: "raises", "TypeError", "NotImplemented", "wrong comparison", "fails when", "incompatible", "edge case", "precision", "overflow", "None", etc.
        - Avoid generic phrases: "doesn't work", "bug", "issue", "unexpected behavior" unless paired with a concrete anchor.
        - Expansions must be diverse: each should target a different failure mode or behavior (comparison semantics, parsing, exception handling, type coercion, boundary conditions, etc.).
        - No markdown, no numbering, no extra commentary.

        Return ONLY a JSON array of strings, like:
        ["expansion 1","expansion 2",...,"expansion {num_queries}"]
    '''
    return prompt


# Let's try with a simple prompt this time
# This one is similar to prompt 2 in structure
# With temp=0.6, MAP@5 is 0.153 and Hit@5 is 0.3 on 10 python repos
# With temp=0.0, MAP@5 is 0.15 and Hit@5 is 0.3 on 10 python repos
# With added "Do not make up hypothetical bugs", MAP@5 is 0.0 and Hit@5 is 0.0 on 10 python repos
def generateQueriesSimple1(snippet: str, num_queries: int, num_tokens: int):
    prompt = f'''
        Given the following code file: 
        <code-snippet>
        {snippet}
        </code-snippet>

        Find up to {num_queries} DISTINCT short bug reports that a developer might write when THIS file is responsible for a bug.
        
        The goal is to create bug reports that would match a real bug report under retrieval.

        Each report should be <= {num_tokens} tokens and include specific code elements from the snippet (method names, class names, exception names, functionality).

        Make it very specific to this code file by including identifiers verbatim.
        Do not make up hypothetical bugs. If there is no bug in the code, return an empty list.

        Return ONLY a JSON array of strings:
        ["query1", "query2", ..., "query{num_queries}"]

        No markdown, no numbering, no extra commentary.
    '''
    return prompt


# This one is a more comprehensive prompt, similar to prompt 3 but for bug report summaries, unconstrained in length
# Resutls with 10 python repos: MAP@5 is 0.07 and Hit@5 is 0.3
def generateQueriesSimple2(snippet: str, num_queries: int, num_tokens: int):
    prompt = f'''
        Given the following code file: 
        <code-snippet>
        {snippet}
        </code-snippet>

        Find up to {num_queries} DISTINCT bug report summaries that a developer might write when THIS file is responsible for a bug.

        The goal is to create bug report summaries that would match a real bug report under retrieval.

        Each bug must include:
            1. Concise problem description: What is the bug or unexpected behavior?

            2. **Key components**: Include identifiers verbatim for buggy classes and functions. Include:
            - Class names, method names involved in the bug
            - Potential problematic interactions with API methods and their parameters
            - Issues with important data structures or state management

            3. **Technical signals**: Explicitly mention any issues with or related to:
            - Dependencies and imports (external libraries, internal modules)
            - Error types thrown or handled (exceptions, error codes)
            - Configuration or environment variables used
            - File I/O operations, database queries, or network calls
            - Exception types that could be raised based on the file context

            4. **Notable bug category details**: Mention bug category, such as algorithms, edge cases, performance issues, security or concurrency problems

            Use technical vocabulary and specific identifiers from the code. Do not make up identifiers or dependencies not present in the file.
            Do not make up hypothetical bugs. If there is no bug in the code, return an empty list.

        Return ONLY a JSON array of strings:
        ["summary1", "summary2", ..., "summary{num_queries}"]

        No markdown, no numbering, no extra commentary.
    '''
    return prompt

# This summary prompt template asks the LLM to include bugs that are not necessarily present in the code, but are plausible.
def generateQueriesSimple3(snippet: str, num_queries: int, num_tokens: int):
    prompt = f'''
        Given the following code file: 
        <code-snippet>
        {snippet}
        </code-snippet>

        Generate {num_queries} DISTINCT bug report summaries that a developer might write when THIS file is responsible for a bug.
        
        The goal is to create bug report summaries that would match a real bug report under retrieval. Include bugs that are plausible given the file context. For instance, if the file has a function that does division, mention division by zero errors, if the code includes database access, cover what could go wrong with that, etc.
        Make it very specific to this code file by including identifiers verbatim.

        Focus on a unique bug category for each bug, for instance: algorithm, database, edge case, performance issue, security or concurrency problem.

        Return ONLY a JSON array of strings:
        ["summary1", "summary2", ..., "summary{num_queries}"]

        No markdown, no numbering, no extra commentary.
    '''
    return prompt