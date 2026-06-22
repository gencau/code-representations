from openai import OpenAI, APIError
import os
import time
import re

class GroqClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['GROQ_API_KEY'], base_url="https://api.groq.com/openai/v1")
        self.model_name = "openai/gpt-oss-20b" #"qwen/qwen3-32b" 
  
    def formatAndSend(self, prompt) -> str:
        messages=[
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": prompt },
        ]
        return self.chat(messages)
    
    def parseResetTime(self, reset_header):
        total = 0.0
        if match := re.search(r'(\d+\.?\d*)m', reset_header):
            total += float(match.group(1)) * 60
        if match := re.search(r'(\d+\.?\d*)s', reset_header):
            total += float(match.group(1))
        return total

    def chat(self, messages):
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0.0,
                    seed=42,
                    messages=messages,
                    stream=False,
                    max_completion_tokens=2048,
                )
                # Remove <think>...</think> tags and their content
                clean_content = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)
                return clean_content
                
            except APIError as e:
                if e.status_code == 429:
                    retry_after = e.response.headers.get('retry-after', base_delay)
                    wait_time = max(float(retry_after), base_delay * (2 ** attempt))
                    print(f"Rate limited. Retry #{attempt+1} in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    continue
            except:
                # any other exception, retry
                continue
                    
        raise RuntimeError(f"Max retries ({max_retries}) exceeded")

 

