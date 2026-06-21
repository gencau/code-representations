import ast
import os
from typing import List, Optional
from lib2to3 import refactor, pytree

from dataclasses import dataclass

@dataclass
class CodeChunk:
    name: str
    type: str  # 'class', 'module' or 'function'
    code: str
    docstring: Optional[str]
    file_path: str

def treeToCode(node):
    """ Converts a lib2to3 tree node to python code string """
    if isinstance(node, pytree.Leaf):
        return node.prefix + node.value
    elif isinstance(node, pytree.Node):
        return "".join(treeToCode(child) for child in node.children)
    else:
        return str(node)
        
def parse_python_text(text: str, file_path: str) -> List[CodeChunk]:
    chunks = []
    # Implement the check here, if we need to use a python2 to 3 converter
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        fixers = ['lib2to3.fixes.fix_print']
        try:
            tool = refactor.RefactoringTool(fixers)

            # Some python code is 2.x, won't parse...
            tree = tool.refactor_string(text, file_path)
            tree = ast.parse(str(tree))
        except:
            print(f"---- {file_path} ---- Won't parse!!!! Returning full text.")
            chunks.append( CodeChunk(
                name=os.path.basename(file_path),
                type='module',
                code=text,
                docstring='',
                file_path=file_path))
            return chunks

    try:
        module_docstring = ast.get_docstring(tree)
    except:
        pass

    # Separate nodes into two groups
    module_nodes = []
    class_func_nodes = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            class_func_nodes.append(node)
        else:
            module_nodes.append(node)

    # Process module-level first
    module_code = '\n'.join([
        ast.get_source_segment(text, n) 
        for n in module_nodes if ast.get_source_segment(text, n)
    ]).strip()
    
    if module_code or module_docstring:
        chunks.append(CodeChunk(
            name=os.path.basename(file_path),
            type='module',
            code=module_code,
            docstring=module_docstring,
            file_path=file_path
        ))

    # Then process classes/functions in original order
    for node in class_func_nodes:
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            class_code = ast.get_source_segment(text, node)
            chunks.append(CodeChunk(
                name=node.name,
                type='class',
                code=class_code,
                docstring=docstring,
                file_path=file_path
            ))
        elif isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            func_code = ast.get_source_segment(text, node)
            chunks.append(CodeChunk(
                name=node.name,
                type='function',
                code=func_code,
                docstring=docstring,
                file_path=file_path
            ))
    
    return chunks


def parse_python_file(file_path: str) -> List[CodeChunk]:
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    return parse_python_text(file_content, file_path)
