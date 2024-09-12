from konlpy.corpus import kolaw
from konlpy.tag import Okt
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from tensorflow import keras
from datetime import datetime
import os
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def clean_str(string):
    try:
        string = re.sub(r"[^가-힝A-Za-z0-9(),!?\'\`]", " ",string)
        string = re.sub(r"\'s", " \'s",string)
        string = re.sub(r"\'ve", " \'ve",string)
        string = re.sub(r"n\'t", " n\'t",string)
        string = re.sub(r"\'re", " \'re",string)
        string = re.sub(r"\'d", " \'d",string)
        string = re.sub(r"\'ll", " \'ll",string)
        string = re.sub(r",", " , ",string)
        string = re.sub(r"!", " ! ",string)
        string = re.sub(r"\(", " \( ",string)
        string = re.sub(r"\)", " \) ",string)
        string = re.sub(r"\?", " \? ",string)
        string = re.sub(r"\s{2,}", " ",string)
    except:
        string = "의"

    return string.lower()

def make_equal_length_solo(matrix, fill_value=""):
    max_length = 25
    if len(matrix) < max_length:
        matrix.extend([fill_value] * (max_length - len(matrix)))
    elif len(matrix) > max_length:
        matrix = row[:max_length]
    return matrix

#테스트
model=keras.models.load_model("C:/Users/minth/PythonProject/PosNegSigmoid2.h5")

okt = Okt()
x = clean_str(input())
stopwords = ["의","가","이","은","들","는","좀","잘","걍","과","도","를","으로","자","에","와","한","하다"]

tokenized_sentence = okt.morphs(x, stem=True)
stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
tokenized_data = stopwords_removed_sentence

test_x = make_equal_length_solo(tokenized_data)
print(np.array(test_x).shape)

word_model = Word2Vec.load("word_model.model")

test_idx_x = []
for i in test_x:
    if i in word_model.wv.key_to_index:
        test_idx_x.append(word_model.wv.key_to_index[i])
    else:
        test_idx_x.append(len(word_model.wv))

prediction = model.predict([test_idx_x])
print(prediction[0])
