import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from random import randint
from sentence_transformers import SentenceTransformer, SentencesDataset, ParallelSentencesDataset,InputExample, evaluation, losses
from model import model
from data_preparation import generate_dataset, shuffle


class ModelParams:
    def __init__(self):
        self.dt = sys.argv[1]
        self.pre_trained_model = sys.argv[2]
        self.mode = sys.argv[3]
        self.epochs = sys.argv[4]
        self.train_path = sys.argv[5]
        self.custom_train_size = sys.argv[6]
        self.custom_eval_size = sys.argv[7]
        self.batch_size = sys.argv[8]
        self.output_dir = sys.argv[9]
        self.loss = losses.CosineSimilarityLoss


def main():
    params = ModelParams()
    for key, value in params.__dict__.items():
        print(key, '=', value)
    if params.mode == 'train':
        start_tome = time.time()
        train_path = params.train_path
        output_path = params.output_dir
        MODEL = model(params)
        LOSS = params.loss(MODEL)
        print('-----generate training data-----')
        st1, st2, label = generate_dataset(params.train_path, column_names_list=['question1','question2','is_duplicate'])
        st1_shuffle, st2_shuffle = shuffle(st1), shuffle(st2)
        train_data = []
        if int(params.custom_train_size):
            TRAIN_SIZE = int(params.custom_train_size)
            EVAL_SIZE = int(params.custom_eval_size)
        else:
            TRAIN_SIZE = int(len(st1) * 0.8)
            EVAL_SIZE = len(st1) - TRAIN_SIZE
        for idx in range(TRAIN_SIZE):
            train_data.append(InputExample(st1[idx], st2[idx], label[idx]))
            # train_data.append(InputExample(st1_shuffle[idx], st2_shuffle[idx], label=0.0))   # comment this line if the negative cases are included
        train_dataset = SentencesDataset(train_data, model=MODEL)
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=int(params.batch_size), num_workers=0)
        st1_eval, st2_eval = st1[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE], st2[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE]
        # st1_eval.extend(list(st1_shuffle[TRAIN_SIZE:])) #comment this line if the negative cases are included
        # st2_eval.extend(list(st2_shuffle[TRAIN_SIZE:])) #comment this line if the negative cases are included
        # scores = [1.0]*EVAL_SIZE+[0.0]*EVAL_SIZE        #comment this line if the negative cases are included
        scores = label[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE]
        print('sentence(train) length: ', len(train_data))
        print('sentence(eval) length: ', len(st1_eval))
        print('scores length: ', len(scores))

        print('-----train-----')
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1=st1_eval, sentences2=st2_eval, scores=scores)
        MODEL.fit(train_objectives=[(train_loader, LOSS)], epochs=int(params.epochs), warmup_steps=100,
                  evaluator=evaluator,
                  evaluation_steps=100, output_path=params.output_dir)


if __name__ == '__main__':
    main()

