import pandas as pd
import numpy as np
from copy import deepcopy
from random import randint
import torch

def generate_dataset(path, column_names_list=['sentence1','sentence2','label']):
    """

    :param path: csv/tsv filepath
    :param column_names_list: the columns to be extracted in the form of list, default= ['sentence1','sentence2','label']
    :return: sentence1, sentence2, label(float) in the form of list
    """
    print(path.split('.')[-1])
    f = []
    if path.split('.')[-1] == 'tsv':
        f = pd.read_csv(path, sep='\t')
    elif path.split('.')[-1] == 'csv':
        f = pd.read_csv(path)

    list_sentence1 = np.array(f[column_names_list[0]]).tolist()
    list_sentence2 = np.array(f[column_names_list[1]]).tolist()
    label = np.array(f[column_names_list[2]], dtype=np.float).tolist()
    return list_sentence1, list_sentence2, label


generate_dataset('/Users/chendi/dataset/nlp_st_paws/train.tsv', ['sentence1','sentence2','label'])


def shuffle(lst):
    temp_list = deepcopy(lst)
    l = len(temp_list)
    while (l):
        l -= 1
        i = randint(0, l)
        temp_list[l], temp_list[i] = temp_list[i], temp_list[l]
    return temp_list