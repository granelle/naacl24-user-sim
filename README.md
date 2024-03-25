# Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation


This is the code for our paper:

**Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation**
Se-eun Yoon, Zhankui He, Jessica Maria Echterhoff, Julian McAuley
Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)

Link to arxiv: https://arxiv.org/abs/2403.09738


### Preparing the dataset

Download the datasets in the `data/common` directory.

* `data/common/ml-25m` should contain csv files downloaded from this [ML-25M repository](https://grouplens.org/datasets/movielens/25m/).
* `data/common/reddit` should contain  `{test, train, valid}.csv` and `id2name.json` files downloaded from this [huggingface repository](https://huggingface.co/datasets/ZhankuiHe/reddit_movie_large_v1).
* `data/redial` should contain `{test_data, train_data}.jsonl` and `movies_with_mentions.csv` files downloaded from this [ReDial repository](https://redialdata.github.io/website/).
* `data/demographic` and `data/imdb` already have their contents in this repository.

Next, preprocess the data with notebooks in `preprocess_notebooks`. Orders don't matter, except that `preprecess_rest.ipynb` should be run last.


### Running the tasks

Copy and paste your OpenAI API key to `openai_key.txt`. We used `openai 0.28.1` in our experiments. 

The code for running each task is in `generate.py` within each folder:

* Task1 (ItemsTalk): `t1_items`
* Task2 (BinPref): `t2_bin_preference`
* Task3 (OpenPref): `t3_open_preference`
* Task4 (RecRequest): `t4_requests`
* Task5 (Feedback): `t5_feedback`

Examples of running this script is in each bash file named `run.sh`.