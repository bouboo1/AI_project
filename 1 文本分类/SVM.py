'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-05 11:27:39
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-21 09:45:27
FilePath: \Project1c:\Users\47517\Desktop\大三上\当代人工智能\code\1\10215501433  仲韦萱 文本分类\SVM.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from sklearn.model_selection import GridSearchCV
import json
import pandas as pd
from sklearn.model_selection import   train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

with open('train_data.txt', 'r') as f:
    train_data = [json.loads(line.strip()) for line in f]

df = pd.DataFrame(train_data)
X = df["raw"]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), params, cv=3, n_jobs=-1)
grid_search.fit(X_train_vec, y_train)
print('Best parameters:', grid_search.best_params_)

clf = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
clf.fit(X_train_vec, y_train)
'''
with open('classifier_SVM.pkl', 'wb') as file:
    pickle.dump(clf, file)
with open('vectorizer_SVM.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
'''
y_pred = clf.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)


'''
Train set size: 6400
Validation set size: 1600
Best parameters: {'C': 10, 'kernel': 'linear'}
Validation Accuracy: 0.955
'''


