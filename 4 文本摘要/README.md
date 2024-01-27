## 实验四：文本摘要

## Setup

详见requirenments.txt

```python
pip install -r requirements.txt
```

## Structure

```
+--data
|      +--test.csv
|      +--train.csv
+--main.py # 运行，包含数据处理、训练和测试画图
+--main_image.py # 运行，包含数据处理、训练和测试（依次跑三种模型画总的bleu图）
+--model_save # 存放预训练后模型参数的目录
+--model  # 存放三个模型代码 
|      +--lstm.py
|      +--gru.py
|      +--rnn.py
+--report.csv # 最终的预测文件
+--requirements.txt
+--image
|      +--bleu_score_all.png三种折线图图
|      +--bleu_score_GRU.png 
|      +--bleu_score_RNN.png 
|      +--bleu_score_LSTM.png 
|      +--loss_curve_GRU.png 
|      +--loss_curve_RNN.png
|      +--loss_curve_LSTM.png
```

## Usage

通过运行 main.py 可以进行模型的训练，main.py 为训练提供了 42个可选参数：

--model: 配置模型
--epoch: 配置 epoch 数

在终端中输入

```
python main.py --epochs 10 --model LSTM 
```
