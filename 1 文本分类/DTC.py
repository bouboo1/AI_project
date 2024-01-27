'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-10-07 22:09:52
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-12 16:24:52
FilePath: \work\DTC.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
# 读取数据集
with open('train_data.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

X = []
y = []

# 解析数据集
for line in data:
    sample = json.loads(line)
    X.append(sample['raw'])
    y.append(sample['label'])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用TF-IDF向量化器将文本映射为向量
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 网格搜索的参数列表
param_grid = {
    'max_depth': [2000,5000],
    'min_samples_leaf': [2,5,10],
    'min_samples_split':[2,20,100,200],
    'criterion': ['gini', 'entropy'],
}

# 网格搜索
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
grid_search.fit(X_train_vec, y_train)

# 输出最佳超参数
print("Best Parameters: ", grid_search.best_params_)

# 在验证集上评估模型性能
classifier = DecisionTreeClassifier(**grid_search.best_params_)
classifier.fit(X_train_vec, y_train)

with open('classifier_DTC.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('vectorizer_DTC.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
y_pred = classifier.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_pred)


# 输出验证集准确率
print("Validation Accuracy: ", accuracy)