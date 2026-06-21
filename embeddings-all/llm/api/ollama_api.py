import re
import subprocess

class ModelParameters:
    def __init__(self, model_name):
        self.model_name = model_name

class Ollama:
    def __init__(self, params : ModelParameters):
        self.model_params = params

    def get_model_response(self, prompt):
        
        command = ["ollama", "run", self.model_params.model_name, prompt]
        answer = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        newcode = ""
        if answer.returncode == 0:
            result = re.search(r'```(.*)```', answer.stdout,re.DOTALL)
            if result:
                newcode = result.group(1)
                print(newcode)
            else:
                newcode = "no code"
                print("no code:")
        
        return newcode