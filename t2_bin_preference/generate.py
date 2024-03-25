import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import json
import sys
sys.path.append('.')
from lib.openai_api import *

DIR = "data/t2_bin_preference"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lm", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--data", type=str, default="all-300")
    parser.add_argument("--anchor", type=str, default="demographic", choices=["demographic", "pickiness"])
    args = parser.parse_args()
    
    save_dir = Path(f"{DIR}/generated/{args.lm}/{args.data}/{args.anchor}/")
    save_dir.mkdir(parents=True, exist_ok=True)  
    
    df = pd.read_csv(f"{DIR}/{args.data}.csv")
    
    with open('data/common/demographic/surnames.txt') as f:
        surnames = f.read()
    surnames = [name.strip() for name in surnames.split(',')]
        
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
    
    for i, row in df.iterrows(): 
        print(f"Movie # {i}")
        
        movie_id = row['movieId']
        title = row['title']
        
        user_responses = []
        for u in tqdm(range(100)):   
            
            prefix = random.choice(['Mr.', 'Ms.'])
            surname = random.choice(surnames)
            prompt = f"Pretend to be {prefix} {surname}."
            
            if args.anchor == "pickiness":    
                pickiness = random.choice(["extremely picky", "moderately picky", "not picky"])
                prompt += f" You are {pickiness} about movies."
             
            prompt += f" You watched the movie {title}. Did you like the movie? Answer Yes or No. Don't say anything else."
            
            response = call_api(prompt, args)
            
            user_responses.append(response)
        
        generated_responses.append(
            {
                'movieId': movie_id,
                'user_responses': user_responses,
                'title': title,
                'prompt': prompt
            }
            )

        # save for every movie
        save_path = f"{save_dir}/responses_{start_idx}-{i+1}.jsonl"
        with open(save_path, 'w') as f:
            for response in generated_responses:
                f.write(json.dumps(response) + '\n')
        print(save_path)
