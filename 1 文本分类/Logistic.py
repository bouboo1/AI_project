import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 读取数据集
with open('train_data.txt', 'r') as f:
    train_data = [json.loads(line.strip()) for line in f]

# 转换为DataFrame格式
df = pd.DataFrame(train_data)

# 按照raw和label分别取出特征和标签
X = df["raw"]
y = df["label"]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))

# 将文本映射成向量（使用TF-IDF）
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 定义参数和参数范围
parameters = {'C':[0.1, 1,10], 'penalty':['l1', 'l2'], 'solver': ['liblinear', 'saga']}

# 创建逻辑回归分类器
classifier = LogisticRegression(max_iter=2000)

# 创建GridSearchCV对象
grid_search = GridSearchCV(classifier, parameters, cv=3, n_jobs=-1)

# 拟合训练数据
grid_search.fit(X_train_vec, y_train)

# 输出最佳参数和准确率
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# 获取最佳模型
best_classifier = grid_search.best_estimator_

# 在验证集上评估性能
y_pred = best_classifier.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy on validation set: ", accuracy)

# 保存模型和向量器
with open('best_classifier.pkl', 'wb') as file:
    pickle.dump(best_classifier, file)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
