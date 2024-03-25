import pandas as pd
from pathlib import Path
import argparse
import ast
from tqdm import tqdm
import random
import json
import sys
sys.path.append('.')
from lib.openai_api import *

DIR = "data/t1_items"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lm", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--target", type=str, default="imdb", choices=["imdb", "reddit", "redial"])
    parser.add_argument("--anchor", type=str, default="demographic", choices=["demographic", "items"])
    args = parser.parse_args()
    
    save_dir = Path(f"{DIR}/generated/{args.lm}/{args.target}/{args.anchor}")
    save_dir.mkdir(parents=True, exist_ok=True)  
    
    df = pd.read_csv(f"{DIR}/{args.target}/processed.csv")      
    
    if args.anchor == "demographic":
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
    
    for i, row in tqdm(df.iterrows(), total=len(df)): 

        test_answer = ast.literal_eval(row['test_answer'])
        n_test = len(test_answer)
        
        if args.anchor == "demographic":
            prefix = random.choice(['Mr.', 'Ms.'])
            surname = random.choice(surnames)
            prompt = f"Pretend to be {prefix} {surname}. You decide to talk about {n_test} movies."
        
        elif args.anchor == "items":
            anchor_str = row['anchor_str']
            
            if args.target == "imdb":
                prompt = f"A person leaves the following remarks on movies...\n{anchor_str}\n...and proceeds to talk about {n_test} more movies."
                
            elif args.target == "reddit":
                utc_time = row['utc_time']
                prompt = f"At UTC time {utc_time}, a person starts to talk about the movies {anchor_str} and proceeds to talk about {n_test} more."
                
            elif args.target == "redial":
                prompt = f"A person mentions {anchor_str} in a conversation about movies and proceeds to mention {n_test} more."
        else:
            raise ValueError
        
        prompt += f" What would these {n_test} movies be? Reply as a list of <Title (yyyy)>. Say nothing else."

        response = call_api(prompt, args)
        
        if response:
            generated_responses.append(
                {
                    'row_i': i,
                    'response': response,
                    'prompt': prompt,
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
    
    
    