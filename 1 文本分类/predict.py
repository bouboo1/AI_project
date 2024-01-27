'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-07 10:04:49
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-12 16:54:07
FilePath: \work\MLP_eval.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pickle

# 打开txt文件并读取内容
with open('test.txt', 'r') as f:
    lines = f.readlines()

#注意用第一个逗号切割
texts = [line.strip().split(", ")[1] for line in lines[1:]]

# 打开保存的向量化器和模型
with open('vectorizer_SVM.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('classifier_SVM.pkl', 'rb') as file:
    classifier = pickle.load(file)
'''
with open('vectorizer_DTC.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('classifier_DTC.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('vectorizer_MLP.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('classifier_MLP.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('vectorizer_Logistic.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('classifier_Logistic.pkl', 'rb') as file:
    classifier = pickle.load(file)
'''


# 转换向量、进行预测
X_test_vec = vectorizer.transform(texts)
y_pred = classifier.predict(X_test_vec)

# 写入文件
lines = ["{}, {}\n".format(i, pred) for i, pred in enumerate(y_pred)]
with open('predict.txt', 'w') as fw:
    fw.write("id, pred\n")
    fw.writelines(lines)







'''
with open('classifier_MLP_SIMPLE.pkl', 'rb') as file:
    classifier = pickle.load(file)
'''