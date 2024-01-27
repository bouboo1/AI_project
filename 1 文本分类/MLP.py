import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# 打开txt文件并读取内容
with open('train_data.txt', 'r') as f:
    train_data = [json.loads(line.strip()) for line in f]

df = pd.DataFrame(train_data)
# 划分特征和标签
X = df["raw"]
y = df["label"]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))

# 实例化TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 将训练集文本转换为向量表示
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 转换为torch tensor
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)

# 设置超参数
input_dim = X_train_tensor.shape[1]
hidden_dim = 110
output_dim = len(set(y_train))

learning_rate = 0.05
epochs = 10

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型、损失和优化器
model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# 训练模型
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

with open('classifier_MLP.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('vectorizer_MLP.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 在验证集上评估模型性能
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val_vec.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.int64)
    predictions = model(X_val_tensor)
    _, predicted = torch.max(predictions, 1)
    accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print("Validation accuracy:", accuracy)

'''
Validation accuracy: 0.924375
'''