# 文本分类

该项目是一个基于逻辑回归、决策树、支持向量机、多层感知机算法的文本分类器。它使用TF-IDF向量化器将文本映射为向量，并通过网格搜索找到最佳超参数，从而实现高效的文本分类。
（**可以选择自行训练模型然后预测，也可以选择使用该文件夹中训练好的模型直接进行预测**）

### 准备数据集

在开始之前，请确保你准备好了：
1. 名为"train_data.txt"”test.txt“和的数据集文件。
"train_data.txt"文件应该包含对象列表，每个对象至少应有"raw"和"label"两个键，用于表示原始文本和标签。格式如下：

| label | raw |
| ----  | ----- | 
| 0  | "I only watche..." | 
| 0 | "I am a huge fan ..." | 
| 1 | "This series revolves..." | 

”test.txt“文件应该包含对象列表，每个对象至少应有"id"和"text"两个键，用于表示编号和文本。格式如下：
| id | text |
| ----  | ----- | 
| 0  | "I only watche..." | 
| 1 | "I am a huge fan ..." | 
| 2 | "This series revolves..." | 




2. 确保安装了所需的库和依赖项：
```
pip install json pandas sklearn pickle
```
并同时确保电脑已经安装python解释器


### 运行代码

1. 首先，进入项目文件夹。
```
 cd c:/Path  #你的项目路径
```
2. 在VSCode的终端中，分别运行SVM.py、DTC.py、 MLP.py、 Logistic.py的代码文件，获得不同分类器对应准确率。
```
python SVM.py
python DTC.py
python MLP.py
python Logistic.py
```
每运行完一个py之后会获得名为”classfier_.pkl“和”vectorizer.pkl“的向量化器和训练好的分类器模型。(文件后缀名为分类器名称，与python文件名一致)

3. 运行预测predict.py文件进行评估（也可以直接运行已经项目中训练好的模型，即文件夹中自带的pkl文件，此时**忽略第1步和第2步**）

```
python predict.py
```
在该文件中选择需要的模型，获得对应模型的预测结果,预测结果会保存在predict.txt文件中
```
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
```
4. 如果想获取TF-IDF和Word2Vec两种向量化器对模型准确率的影响，可以运行
```
python TF_IDF_compare.py
python Word2Vec.py
```


### 注意事项


- 如果需要修改超参数搜索范围，你可以在SVM.py、DTC.py、 MLP.py、 Logistic.py代码中修改"param_grid"变量的值。
- 可以根据项目需求自定义训练和测试数据集，并相应修改文件名和路径。
- 文件result.txt是根据train_data.txt训练，用test.txt进行测试得到的预测结果。
