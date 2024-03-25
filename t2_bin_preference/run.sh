#!/bin/bash

lms=("text-davinci-003" "gpt-3.5-turbo" "gpt-4")
datas=("frequent" "infrequent")
anchors=("demographic" "pickiness")

for lm in "${lms[@]}"; do
  for data in "${datas[@]}"; do
    for anchor in "${anchors[@]}"; do
      python t2_bin_preference/generate.py --lm "$lm" --data "$data" --anchor "$anchor"
      echo "t2_bin_preference/generate.py --lm $lm --data $data --anchor $anchor"
    done
  done
done
