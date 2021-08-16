import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from random import randint
from sentence_transformers import SentenceTransformer, SentencesDataset, ParallelSentencesDataset,InputExample, evaluation, losses

from data_preparation import generate_dataset,shuffle

PATH = '/Users/chendi/dataset/nlp_st_paws/train.tsv'
TRAIN_SIZE = int(49401*0.8)
EVAL_SIZE = int(49401*0.2)
BATCH_SIZE = 20


def train(path, pre_trained_model):
    MODEL = SentenceTransformer(pre_trained_model)
    LOSS = losses.CosineSimilarityLoss(MODEL)
    st1, st2, label = generate_dataset(path)
    st1_shuffle, st2_shuffle = shuffle(st1), shuffle(st2)
    train_data = []
    for idx in range(TRAIN_SIZE):
        train_data.append(InputExample(st1[idx], st2[idx], label=label[idx]))
        train_data.append(InputExample(st1_shuffle[idx],st2_shuffle[idx], label=0.0))

    sentence1_eval = st1[BATCH_SIZE:]
    sentence2_eval = st2[BATCH_SIZE:]
    sentence1_eval.extend(list(st1_shuffle[BATCH_SIZE:]))
    sentence2_eval.extend(list(st2_shuffle[BATCH_SIZE:]))
    scores = [1.0]*EVAL_SIZE+[0.0]*EVAL_SIZE

    train_dataset = SentencesDataset(train_data, model=MODEL)
    train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=BATCH_SIZE)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=st1, sentences2=st2, scores=scores)
    MODEL.fit(train_objectives=[(train_dataloader, LOSS)], epochs=0, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=100, output_path='/Users/chendi/dataset/nlp_st_paws/res')

train('/Users/chendi/dataset/nlp_st_paws/train.tsv', pre_trained_model='paraphrase-distilroberta-base-v1')
path = '/Users/chendi/dataset/nlp_st_paws/train.tsv'
pre_trained_model = 'paraphrase-distilroberta-base-v1'
MODEL = SentenceTransformer(pre_trained_model)
"""
LOSS = losses.CosineSimilarityLoss(MODEL)
st1, st2, label = generate_dataset(path)
st1_shuffle, st2_shuffle = shuffle(st1), shuffle(st2)
train_data = []
for idx in range(TRAIN_SIZE):
    train_data.append(InputExample(st1[idx], st2[idx], label=label[idx]))
    train_data.append(InputExample(st1_shuffle[idx], st2_shuffle[idx], label=0.0))

sentence1_eval = st1[BATCH_SIZE:]
sentence2_eval = st2[BATCH_SIZE:]
sentence1_eval.extend(list(st1_shuffle[BATCH_SIZE:]))
sentence2_eval.extend(list(st2_shuffle[BATCH_SIZE:]))
scores = [1.0] * EVAL_SIZE + [0.0] * EVAL_SIZE

train_dataset = SentencesDataset(train_data, model=MODEL)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=st1, sentences2=st2, scores=scores)
MODEL.fit(train_objectives=[(train_dataloader, LOSS)], epochs=0, warmup_steps=100, evaluator=evaluator,evaluation_steps=100, output_path='/Users/chendi/dataset/nlp_st_paws/res')

"""