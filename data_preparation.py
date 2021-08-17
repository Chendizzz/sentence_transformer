import pandas as pd
import numpy as np
from copy import deepcopy
from random import randint
from sentence_transformers import SentenceTransformer
import scipy


# these lines are some interesting tests on the pre-trained model
"""
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
sentence1 = ['I am Groot',
             'I like eating apples',
             'I do not like women']
sentence2 = ['I am a tree',
             'I enjoy eating apples',
             'I am a noob']
label = [1.0,1.0,1.0]

data = {'sentence1': sentence1, 'sentence2': sentence2, 'label': label}
df = pd.DataFrame(data)
print(data)
df.to_csv('/Users/chendi/dataset/nlp_st_paws/train_.csv')

sentence = sentence1+sentence2
sentence_embeddings = model.encode(sentence)

for sentence_, embeddings in zip(sentence,sentence_embeddings):
    print('sentence: ', sentence_)
    print('embeddinges: ', embeddings)
    print('')

query = ['I hate women']
query_embeddings = model.encode(query)
num_of_top_matches = 2
for query, query_embeddings in zip(query, query_embeddings):
    distance = scipy.spatial.distance.cdist([query_embeddings], sentence_embeddings, 'cosine')[0]
    res = zip(range(len(distance)), distance)
    res = sorted(res, key=lambda x : x[1])
    print('query: ', query)
    print('\nTop{} most similar sentence in corpus: '.format(num_of_top_matches))
    print(res)
    for idx, distance in res[0:num_of_top_matches]:
        print('matched sentece: ', sentence[idx]+'//' 'cosine score: ', (1-distance))
"""

def generate_dataset(path, column_names_list):
    """

    :param path: csv/tsv filepath
    :param column_names_list: the columns to be extracted in the form of list, default= ['sentence1','sentence2','label']
    :return: sentence1, sentence2, label(float) in the form of list
    """
    f = []
    print(path.split('.')[-1])
    if path.split('.')[-1] == 'tsv':
        f = pd.read_csv(path, sep='\t')
    elif path.split('.')[-1] == 'csv':
        f = pd.read_csv(path)

    list_sentence1 = np.array(f[column_names_list[0]]).tolist()
    list_sentence2 = np.array(f[column_names_list[1]]).tolist()
    label = np.array(f[column_names_list[2]], dtype=np.float).tolist()
    print('length: ', len(list_sentence1))
    print('example: ', list_sentence1[0]+'//'+list_sentence2[0])
    return list_sentence1, list_sentence2, label


#generate_dataset('/Users/chendi/dataset/QQP/train.tsv', ['question1','question2','is_duplicate'])



def shuffle(lst):
    temp_list = deepcopy(lst)
    l = len(temp_list)
    while (l):
        l -= 1
        i = randint(0, l)
        temp_list[l], temp_list[i] = temp_list[i], temp_list[l]
    return temp_list
