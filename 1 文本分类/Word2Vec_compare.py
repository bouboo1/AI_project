'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-05 11:34:20
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-05 11:43:11
FilePath: \work\Word2Vec.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import random
import numpy as np
from sklearn.svm import SVC
import nltk
import gensim


# 定义分词函数
def tokenize(text):
    return nltk.word_tokenize(text.lower())


# 定义Word2Vec模型训练函数
def train_word2vec(sentences):
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    return model


# 定义文本向量化函数
def encode_text(text, model):
    words = tokenize(text)
    vector = []
    for word in words:
        if word in model.wv.vocab:
            vector.append(model.wv[word])
    if not vector:
        return [0] * 100
    return np.mean(vector, axis=0)


# 读入数据
with open('train_data.txt', 'r') as f:
    data = [json.loads(line) for line in f]

# 划分训练集和测试集
random.shuffle(data)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 训练Word2Vec模型
sentences = [tokenize(d['raw']) for d in train_data]
model = train_word2vec(sentences)

# 将每个文本向量化
train_features = [encode_text(d['raw'], model) for d in train_data]
test_features = [encode_text(d['raw'], model) for d in test_data]

# 训练SVM模型
svm = SVC(kernel='linear')
svm.fit(train_features, [d['label'] for d in train_data])

# 在测试集上评估模型性能
acc = svm.score(test_features, [d['label'] for d in test_data])
print(f" Word2Vec Test Accuracy: {acc:.4f}")