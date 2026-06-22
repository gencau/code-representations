"""
    Test conversion of some python 2.x files to 3.x for parser.
"""
import ast
from lib2to3 import refactor

# Create a fixer for Python 2-to-3 syntax
fixers = ['lib2to3.fixes.fix_print']
tool = refactor.RefactoringTool(fixers)

filename = 'data/thealgorithms_python_dbfc220264e514cbc94320b6e4769acfaea85fad/ciphers/Onepad_Cipher.py'
# Read the Python 2 code
with open(filename, 'r', encoding='utf-8') as f:
    code = f.read()

# Apply fixes (e.g., convert `print "Hello"` to `print("Hello")`)
tree = tool.refactor_string(code, filename)

# Now parse the fixed code with `ast`
parsed_ast = ast.parse(str(tree))
print(ast.dump(parsed_ast))