#!/bin/bash

lms=("text-davinci-003" "gpt-3.5-turbo" "gpt-4")

for lm in "${lms[@]}"; do
  python t2_bin_preference/generate.py --lm "$lm" 
  echo "t2_bin_preference/generate.py --lm $lm
done
