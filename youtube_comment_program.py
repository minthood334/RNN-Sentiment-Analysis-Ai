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

driver = webdriver.Chrome()

# 유튜브 동영상 페이지로 이동
video_url = "https://youtu.be/nUSaYpoPoBM?si=WyJP_9YtFe7uVXYJ"
driver.get(video_url)

# 스크롤을 아래로 내려 댓글을 로드
last_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# 댓글 요소들 가져오기
comment_elements = driver.find_elements(By.CSS_SELECTOR, "#content #content-text")

comments = []
for comment_element in comment_elements:
    comments.append(comment_element.text)

# 웹드라이버 종료
driver.quit()

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

model=keras.models.load_model("C:/Users/minth/PythonProject/PosNegSigmoid2.h5")

okt = Okt()
line_x = [clean_str(i) for i in comments]
stopwords = ["의","가","이","은","들","는","좀","잘","걍","과","도","를","으로","자","에","와","한","하다"]

tokenized_data = []
for sentence in line_x:
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    tokenized_data.append(stopwords_removed_sentence)

def make_equal_length(matrix, fill_value=""):
    max_length = 25
    for i, row in enumerate(matrix):  # enumerate를 사용하여 인덱스와 함께 순회
        if len(row) < max_length:
            matrix[i].extend([fill_value] * (max_length - len(row)))
        elif len(row) > max_length:
            matrix[i] = row[:max_length]
    return matrix
train_x = make_equal_length(tokenized_data)
print(np.array(train_x).shape)

word_model = Word2Vec.load("word_model.model")

train_idx_x = []
for i in train_x:
    line = []
    for j in i:
        if j in word_model.wv.key_to_index:
            line.append(word_model.wv.key_to_index[j])
        else:
            line.append(len(word_model.wv))
    train_idx_x.append(line)

prediction = model.predict(train_idx_x)
sumpre = sum(prediction) / len(comments)

'''
if sumpre >= 0.60:
    print("매우 긍정적")
elif sumpre >= 0.50:
    print("긍정적")
elif sumpre >= 0.40:
    print("대체로 긍정적")
elif sumpre >= 0.30:
    print("복합적")
elif sumpre >= 0.20:
    print("대체로 부정적")
elif sumpre >= 0.10:
    print("부정적")
else:
    print("매우 부정적")
'''
print(sumpre)
