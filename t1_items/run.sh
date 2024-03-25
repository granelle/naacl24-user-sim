#!/bin/bash

lms=("text-davinci-003" "gpt-3.5-turbo" "gpt-4")
targets=("imdb" "reddit" "redial")
anchors=("demographic" "items")

for lm in "${lms[@]}"; do
  for target in "${targets[@]}"; do
    for anchor in "${anchors[@]}"; do
      python t1_items/generate.py --lm "$lm" --target "$target" --anchor "$anchor"
      echo "t1_items/generate.py --lm $lm --target $target --anchor $anchor"
    done
  done
done