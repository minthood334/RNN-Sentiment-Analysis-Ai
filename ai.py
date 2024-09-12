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

okt = Okt()

file_url = 'C:/Users/minth/OneDrive/바탕 화면/감정분석(230816)/dataset/*.txt'
file_path = glob(file_url)
comment = pd.read_csv("C:/Users/minth/인기급상승 라벨 담.csv", encoding='UTF8')

x=[]
y=[]
for path in file_path:
    file = open(path, encoding='UTF-8')
    for i, line in enumerate(file):
        if i != 0:
            line = line.split()
            x.append(' '.join(line[1:-1]))
            y.append(int(line[-1]))
    file.close()
x = np.concatenate((np.array(x), comment["Comment"].values))
y = np.concatenate((np.array(y), comment["Label"].values))

shuffle = np.arange(len(x))
np.random.shuffle(shuffle)

x = x[shuffle]
y = y[shuffle]

#토그나이징
line_x = [clean_str(i) for i in x]
stopwords = ["의","가","이","은","들","는","좀","잘","걍","과","도","를","으로","자","에","와","한","하다"]

tokenized_data = []
for sentence in line_x:
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    tokenized_data.append(stopwords_removed_sentence)

#쓰기
import pickle
with open('tokenized_data_file2.pkl', 'wb') as f:
    pickle.dump(tokenized_data, f)
with open('y_data_file2.pkl', 'wb') as f:
    pickle.dump(y, f)

#불러오기
import pickle
tokenized_load = []
with open('tokenized_data_file.pkl', 'rb') as f:
    tokenized_data = pickle.load(f)
with open('y_data_file.pkl', 'rb') as f:
    y = pickle.load(f)

#학습 데이터 분류
train_idx = int(len(tokenized_data) * 0.8)
train_x, train_y = tokenized_data[:train_idx], y[:train_idx]
test_x, test_y = tokenized_data[train_idx:], y[train_idx:]

#워드벡터 모델 생성 
word_model = Word2Vec(sentences = tokenized_data, vector_size=200, window = 5, min_count = 5, workers = 10)
word_model.save('word_model.model')

#모든 벡터를 같은 크기로 지정
def make_equal_length(matrix, fill_value=""):
    max_length = 25
    for i, row in enumerate(matrix):  # enumerate를 사용하여 인덱스와 함께 순회
        if len(row) < max_length:
            matrix[i].extend([fill_value] * (max_length - len(row)))
        elif len(row) > max_length:
            matrix[i] = row[:max_length]
    return matrix
tokenized_data = make_equal_length(tokenized_data)

word_model = Word2Vec.load("word_model.model")
#인덱스 배열 생성
idx_x = []
for i in tokenized_data:
    line = []
    for j in i:
        if j in word_model.wv.key_to_index:
            line.append(word_model.wv.key_to_index[j])
        else:
            line.append(-1)
    idx_x.append(line)
#훈련 데이터와 테스트 데이터 나누기
train_idx = int(len(idx_x) * 0.8)
train_x, train_y = idx_x[:train_idx], y[:train_idx]
test_x, test_y = idx_x[train_idx:], y[train_idx:]

#임베딩 하기
embedding_matrix = np.zeros((len(word_model.wv) + 1, 200))
def get_vector(word):
    if word in word_model.wv.key_to_index:
        return word_model.wv[word]
    else:
        return None

for i in tokenized_data:
    for j in i:
        vector_value = get_vector(j)
        if vector_value is not None:
            embedding_matrix[word_model.wv.key_to_index[j]] = vector_value

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_model.wv) + 1, 200, weights=[embedding_matrix], input_length=25, trainable=False),
    tf.keras.layers.GRU(units=50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.summary()

tx = np.array(train_x)
ty = np.array(train_y)

history = model.fit(tx, ty, epochs=10, batch_size=128, validation_split=0.2)

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

model.save("C:/Users/minth/PythonProject/PosNegSigmoid2.h5")
