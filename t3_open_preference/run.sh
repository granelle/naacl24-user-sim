#!/bin/bash

lms=("text-davinci-003" "gpt-3.5-turbo" "gpt-4")
anchors=("demographic" "pickiness")

for lm in "${lms[@]}"; do
  for anchor in "${anchors[@]}"; do
    python t2_bin_preference/generate.py --lm "$lm" --anchor "$anchor"
    echo "t2_bin_preference/generate.py --lm $lm --anchor $anchor"
  done
done
