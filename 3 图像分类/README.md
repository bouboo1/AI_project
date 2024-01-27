## 实验三：图像分类及经典CNN实现（MNIST）

## Setup

详见requirenments.txt
```python
pip install -r requirements.txt
```

## Structure

```
|-- main.py #数据处理、运行脚本
|-- README.md   #对仓库的解释
|-- requirements.txt    #本次实验的环境
|-- LeNet  #LeNet模型
|-- AlexNet  #AlexNet模型
|-- VGGNet  #LeNet模型
|-- RetNet  #LeNet模型
|-- MobileNet  #LeNet模型
|-- DenseNet  #LeNet模型
```

## Usage
通过运行 main.py 可以进行模型的训练，main.py 为训练提供了 4 个可选参数：

--model: 配置模型
--lr: 配置学习率
--batch_size: 配置 batchsize
--epoch: 配置 epoch 数

在终端中输入

```
python main.py --model LeNet --lr 0.001 --batch_size 128 --epoch 10
```

另，本次实验使用cpu设备。且运行时间较长。