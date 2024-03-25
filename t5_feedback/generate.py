import pandas as pd
from pathlib import Path
import argparse
import time
import json
from tqdm import tqdm
import random
import sys
sys.path.append('.')
from lib.openai_api import *

DIR = "data/t5_feedback"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--lm", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--rec", type=str, default="items", choices=["items", "context"])
    parser.add_argument("--action", type=str, default="reject", choices=["reject", "compare"])
    parser.add_argument("--ask_why", action='store_true', help='ask why such feedback was given')
    args = parser.parse_args()
    
    save_dir = Path(f"{DIR}/generated/{args.lm}/{args.rec}/{args.action}")
    if args.ask_why: save_dir /= "ask_why"
    save_dir.mkdir(parents=True, exist_ok=True)     
    
    df = pd.read_csv(f"{DIR}/{args.rec}.csv") 
    
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
        
        req = row["request"]
        pos = row['first']
        neg = row['random']
        
        if args.action == "reject":
            instr  = "In the following conversation, a USER asks for movie recommendations. Your task is to act like the USER by giving the following responses to the AGENT's recommendation:\n"
            instr += "If the recommendation is coherent to your request, answer Accept.\n"
            instr += "If the recommendation is incoherent to your request, answer Reject.\n"
            instr += "Simply answer Accept or Reject.\n"
            if args.ask_why:
                instr += "Provide a short reason (less than 40 words) for your response."

            prompt_pos = instr + f"USER: {req}\nAGENT: {pos}\nUSER (answer Accept or Reject): "
            prompt_neg = instr + f"USER: {req}\nAGENT: {neg}\nUSER (answer Accept or Reject): "
            
            response_to_pos = call_api(prompt_pos, args)
            time.sleep(0.2)
            response_to_neg = call_api(prompt_neg, args)
            
            generated_responses.append({
                'req': req,
                'pos': pos,
                'neg': neg,
                "response_to_pos": response_to_pos,
                "response_to_neg": response_to_neg,
            })
        
        elif args.action == "compare":
            
            order = ["p", "n"]
            random.shuffle(order)
            
            if order[0] == "p":
                a1_response = pos
                a2_response = neg
            else:
                a1_response = neg
                a2_response = pos
            
            prompt = "A USER asks for movie recommendation. AGENT 1 and AGENT 2 gives recommendations. Your task is to choose the AGENT that gives better recommendations. Simply answer AGENT 1 or AGENT 2. You HAVE to choose one."
            prompt += f"USER: {req}\nAGENT 1's response: {a1_response}\nAGENT 2's response: {a2_response}\n\nWhich response is better? (Reply AGENT 1 or AGENT 2)" 
            if args.ask_why:
                prompt += "\nProvide a short reason (less than 40 words) for your response."
              
            response = call_api(prompt, args)
            
            if order[0] == "p":
                correct = "AGENT 1"
            else:
                correct = "AGENT 2"
                
            generated_responses.append({
                'req': req,
                'pos': pos,
                'neg': neg,
                "response": response,
                "correct": correct,
            })
            
        else:
            raise ValueError
        
        
        if args.action == 'reject':
            resp1 = response_to_pos
            resp2 = response_to_neg
        else:
            resp1 = resp2 = response
    
        if not resp1 or not resp2 or (i % 500 == 0 and i > start_idx) or i == end_idx-1:
            save_path = f"{save_dir}/responses_{start_idx}-{i+1}.jsonl"
            with open(save_path, 'w') as f:
                for response in generated_responses:
                    f.write(json.dumps(response) + '\n')
            print(save_path)

        if not resp1 or not resp2:
            break



