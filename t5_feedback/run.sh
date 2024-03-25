#!/bin/bash
lms=("text-davinci-003" "gpt-3.5-turbo" "gpt-4")
recs=("items" "context")
actions=("reject" "compare")

for lm in "${lms[@]}"; do
    for rec in "${recs[@]}"; do
        for action in "${actions[@]}"; do
            echo "t5_feedback/generate.py --lm $lm --rec $rec --action $action"
            python t5_feedback/generate.py --lm "$lm" --rec "$rec" --action "$action"
            # python t5_feedback/generate.py --lm "$lm" --rec "$rec" --action "$action" --ask_why --end 20
        done
    done
done