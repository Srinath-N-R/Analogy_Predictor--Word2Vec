import gensim
import pandas as pd
from gensim.models import KeyedVectors
import os
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

path_glove = os.path.abspath('glove.twitter.27B/glove.twitter.27B.200d.txt')
path_w2v = os.path.abspath('glove.twitter.27B/glove.twitter.27B.200d_w2v.txt')
glove_file = datapath(path_glove)
tmp_file = get_tmpfile(path_w2v)
_ = glove2word2vec(glove_file, tmp_file)
path = os.path.abspath('/content/glove.twitter.27B.200d_w2v.txt')
model = KeyedVectors.load_word2vec_format(path, binary=False)


def display(obj):
    print(pd.DataFrame(obj))

def analogy(word):
    result = model.most_similar(positive=[word])
    return result[0][0]

a = list(str(input('Enter any sentence to find analogy: ')))
a = a[0].split()

predictions = [analogy(word) for word in a]
display(zip(a, predictions))