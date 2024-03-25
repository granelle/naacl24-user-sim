import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.append('.')
from lib.openai_api import *

DIR = 'data/t4_requests'

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lm", type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()
    
    save_dir = Path(f'{DIR}/generated/{args.lm}')
    save_dir.mkdir(exist_ok=True, parents=True)    

    df = pd.read_csv(f"{DIR}/processed.csv")
    
    start_idx = args.start
    end_idx = args.end  # not inclusive

    if not start_idx:
        start_idx = 0
    if not end_idx:
        end_idx = len(df)
        
    assert(start_idx >=0 and end_idx <= len(df))

    lm_to_function = {
        'text-davinci-003': call_api_completion,
        'gpt-3.5-turbo': call_api_chatcompletion,
        'gpt-4': call_api_chatcompletion,
    }
    call_api = lm_to_function[args.lm]

    generated_responses = []
    
    df = df.iloc[start_idx:end_idx]

    for i, row in tqdm(df.iterrows(), total=len(df)): 

        movies_str = row['movies_str']
        target_len = len(row['request'])

        prompt = f"Generate a movie recommendation request. Include (but do not request) the following movies in your text: {movies_str}. Make sure the length of the request is approximately {target_len} characters."

        response = call_api(prompt, args)

        if response:
            generated_responses.append(
                {
                    'response': response,
                    'movies_str': movies_str,
                    'actual_char_len': len(response),
                    'target_char_len': target_len
                }
                )

        if response == None or (i % 1000 == 0 and i > start_idx) or i == end_idx-1:
            save_path = f"{save_dir}/responses_{start_idx}-{i+1}.jsonl"
            with open(save_path, 'w') as f:
                for response in generated_responses:
                    f.write(json.dumps(response) + '\n')
            print(save_path)

        if response == None:
            break