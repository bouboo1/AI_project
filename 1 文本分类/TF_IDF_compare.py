'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-14 10:45:13
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-12 13:57:41
FilePath: \work\spilt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# 打开txt文件并读取内容
with open('train_data.txt', 'r') as f:
    train_data = [json.loads(line.strip()) for line in f]

df = pd.DataFrame(train_data)
# 划分特征和标签
X = df["raw"]
y = df["label"]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# 划分测试集和验证集
train, test = train_test_split(train_data, test_size=0.2, random_state=42)

# 打印测试集和验证集的长度
print("训练集大小:", len(train))
print("测试集大小:", len(test))
'''

print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))

# 实例化TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 将训练集文本转换为向量表示
X_train_vec = vectorizer.fit_transform(X_train)

# 实例化SVM分类器
clf = SVC()

# 训练模型
clf.fit(X_train_vec, y_train)


# 将验证集文本转换为向量表示
X_val_vec = vectorizer.transform(X_val)

# 预测验证集数据
y_pred = clf.predict(X_val_vec)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print("TF-IDF Validation Accuracy:", accuracy)
