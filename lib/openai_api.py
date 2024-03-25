import pandas as pd
from pathlib import Path
import openai
import time

with open('./openai_key.txt', 'r') as f:
    openai.api_key = f.read()

def call_api_chatcompletion(prompt, args):
    
    model_name = args.lm    
    temperature = getattr(args, 'temp', None)    
    
    request_timeout = 20
    for attempt in range(5):  # 5 attempts
        try:
            params = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "max_tokens": 1000,
                "request_timeout": request_timeout
            }

            if temperature is not None:
                params["temperature"] = temperature

            response = openai.ChatCompletion.create(**params)

            return response['choices'][0]['message']['content']
        
        except (openai.error.APIError, openai.error.Timeout, openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            if attempt >= 4:  
                return None
            else:
                print(f"{str(e)}, retrying...")
                time.sleep(min(1 * 2**attempt, 60))  # exponential backoff with max delay of 60 seconds
                request_timeout = min(300, request_timeout * 2)  # increase timeout for next request
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
        
        
def call_api_completion(prompt, args):
    
    model_name = args.lm    
    # temperature = args.temp  # use default temperature
    
    request_timeout = 20
    for attempt in range(5):  # 5 attempts
        try:
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                # temperature=temperature,
                max_tokens=1000,
                request_timeout=request_timeout
            )
            
            return response['choices'][0]['text'].lstrip()
        
        except (openai.error.APIError, openai.error.Timeout, openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            if attempt >= 4:  
                return None
            else:
                print(f"{str(e)}, retrying...")
                time.sleep(min(1 * 2**attempt, 60))  # exponential backoff with max delay of 60 seconds
                request_timeout = min(300, request_timeout * 2)  # increase timeout for next request
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None


