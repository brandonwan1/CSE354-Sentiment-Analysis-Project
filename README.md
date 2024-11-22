# CSE354-Project
NLP Project for James You, Wenjun Yang, Brandon Wan

# Codebase Source
Comes from original Assignment 2 code

# Files Modified
sequence_to_vector.py includes the new models used in this assigment as separate classes.

The following files were changed to allow for the new models to parsed as arguments: \
train.py\
plot_performance_against_data_size.py\
plot_probing_performances_on_sentiment_task.py\
plot_probing_performances_on_bigram_order_task.py\
plot_perturbation_analysis.py

# Commands
Note: Most of these commands are copied straight from the original hw2 README, and are included for completeness/convenience.
## Installation
This assignment is implemented in python 3.6 and torch 1.9.0. Follow these steps to setup your environment:

1. [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-hw2 python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw2
```
4. Install the requirements:
```
pip3 install -r requirements.txt
```

5. Download GloVe wordvectors:
```
./download_glove.sh
```
6. Download Spacy package
```
python3 -m spacy download en_core_web_sm
```

#### Train:

To train the model, the same command format is used. 

i.e.

```
python3 train.py main \
                  data/imdb_sentiment_train_5k.jsonl \
                  data/imdb_sentiment_dev.jsonl \
                  --seq2vec-choice dan \
                  --embedding-dim 50 \
                  --num-layers 4 \
                  --num-epochs 5 \
                  --suffix-name _dan_5k_with_emb \
                  --pretrained-embedding-file data/glove.6B.50d.txt
```

To run the new configurations, use `cnn` for CNN, `danwithattention` for DAN with Attention, `bilstm` for bidirectional LSTM.

#### Predict:

Once the model is trained, you can use its serialization directory and any dataset to make predictions on it. For example, the following command:

```
python3 predict.py serialization_dirs/main_dan_5k_with_emb \
                  data/imdb_sentiment_test.jsonl \
                  --predictions-file my_predictions.txt
```
makes prediction on `data/imdb_sentiment_test.jsonl` using trained model at `serialization_dirs/main_dan_5k_with_emb` and stores the predicted labels in `my_predictions.txt`.

In case of the predict command, you do not need to specify what model type it is. This information is stored in the serialization directory.

#### Evaluate:

Once the predictions are generated you can evaluate the accuracy by passing the original dataset path and the predictions. For example:

```
python3 evaluate.py data/imdb_sentiment_test.jsonl my_predictions.txt
```

#### Probing

Once the `main` model is trained, we can use its frozen representations at certain layer and learn a linear classifier on it by *probing* it. This essentially checks if the representation in the given layer has enough information (extractable by linear model) for the specific task.

To train a probing model, we would again use `train.py`. For example, to train a probing model at layer 3, with base `main` model stored at `serialization_dirs/main_dan_5k_with_emb` on sentiment analysis task itself, you can use the following command:

```
python3 train.py probing \
                  data/imdb_sentiment_train_5k.jsonl \
                  data/imdb_sentiment_dev.jsonl \
                  --base-model-dir serialization_dirs/main_dan_5k_with_emb \
                  --num-epochs 5 \
                  --layer-num 3
```

Similarly, you can also probe the same model on bigram-classification task by just replacing the datasets in the above command.

### Analysis & Probing Tasks:

There are four scripts in the code that will allow you to do analyses on the sentence representations:
1. `plot_performance_against_data_size.py`
2. `plot_probing_performances_on_sentiment_task.py`
3. `plot_probing_performances_on_bigram_order_task.py`
4. `plot_perturbation_analysis.py`

## Software Used
Python 3.6\
Pytorch 1.9.0\
numpy 1.19.5\
tqdm 4.62.2\
mypy 0.910\
mypy-extensions 0.4.3\
typing-extensions 3.10.0.2\
overrides 6.1.0\
spacy 3.1.2\
matplotlib 3.3.4\
tensorboard 2.6.0
