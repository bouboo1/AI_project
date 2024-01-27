import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import os
from model.rnn import RNN
from model.lstm import LSTM
from model.gru import GRU
import argparse
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 将索引序列转换为单词序列
def sequence_to_text(sequence, index_to_word):
    return ' '.join([index_to_word[index] for index in sequence if index != 0])

# 预测并计算 BLEU 分数
def compute_bleu_score(predicted_sequence, target_sequence, index_to_word):
    # 预测目标序列
    # predicted_sequence = model.predict([input_sequence, target_sequence],use_multiprocessing = True)
    predicted_sequence = np.argmax(predicted_sequence, axis=-1)
    
    # 将预测的序列和目标序列转换为单词序列
    predicted_text = sequence_to_text(predicted_sequence[0], index_to_word)
    target_text = sequence_to_text(target_sequence[0], index_to_word)
    
    # 计算 BLEU 分数
    bleu_score = sentence_bleu([target_text.split()], predicted_text.split())
    
    return bleu_score

# 将输入和输出转换为索引序列
def pad(data, word_to_index, max_len):
    sequences = [[word_to_index[word] for word in sentence.split()] for sentence in data]
    return pad_sequences(sequences, maxlen=max_len, padding='post')

# 读取数据集
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 0.25比例划分训练集和验证集
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

max_input_len = max(train_data['description'].map(lambda x: len(x.split())))
max_output_len = max(train_data['diagnosis'].map(lambda x: len(x.split())))

# 构建词汇表
input_vocab = set(' '.join(train_data['description']).split())
output_vocab = set(' '.join(train_data['diagnosis']).split())

# 进行填充，特殊标记
input_vocab_size = len(input_vocab) + 1
output_vocab_size = len(output_vocab) + 1

# 创建输入和输出的索引映射字典
# 索引号从1开始，而不是从0开始，这是因为在后续模型训练过程中，通常会使用0作为填充值的索引
input_word_to_index = {word: index + 1 for index, word in enumerate(input_vocab)}
input_index_to_word = {index + 1: word for index, word in enumerate(input_vocab)}

output_word_to_index = {word: index + 1 for index, word in enumerate(output_vocab)}
output_index_to_word = {index + 1: word for index, word in enumerate(output_vocab)}


# 进行数据处理
train_encoder_input = pad(train_data['description'], input_word_to_index, max_input_len)
train_decoder_input = pad(train_data['diagnosis'], output_word_to_index, max_output_len)

val_encoder_input = pad(val_data['description'], input_word_to_index, max_input_len)
val_decoder_input = pad(val_data['diagnosis'], output_word_to_index, max_output_len)

train_decoder_output = pad(train_data['diagnosis'], output_word_to_index, max_output_len)
val_decoder_output = pad(val_data['diagnosis'], output_word_to_index, max_output_len)


test_encoder_input = pad(test_data['description'], input_word_to_index, max_input_len)
test_decoder_input = pad(test_data['diagnosis'], output_word_to_index, max_output_len)


# 构建Seq2Seq模型
latent_dim = 64

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
# parser.add_argument('--model', default='LSTM')

args = parser.parse_args()

models = ["LSTM", "GRU", "RNN"]
# 定义三个空列表用于保存不同模型的BLEU和Loss
lstm_bleu_scores = []
gru_bleu_scores = []
rnn_bleu_scores = []

lstm_train_losses = []
gru_train_losses = []
rnn_train_losses = []

lstm_val_losses = []
gru_val_losses = []
rnn_val_losses = []



for model_type in models:
    if model_type == "LSTM":
            model = LSTM (max_input_len,max_output_len,input_vocab_size,output_vocab_size,latent_dim)
    elif model_type == "GRU":
            model = GRU(max_input_len,max_output_len,input_vocab_size,output_vocab_size,latent_dim)
    else:
            model = RNN(max_input_len,max_output_len,input_vocab_size,output_vocab_size,latent_dim)

    bleu_scores = []  # 用于保存每个epoch的BLEU分数
    rouge_scores = []  # 用于保存每个epoch的ROUGE分数
    train_losses = [] # 用于保存每个epoch的训练集loss
    val_losses = []   # 用于保存每个epoch的验证集loss
    # 定义回调函数保存最佳模型
    checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

    # 训练模型
    batch_size = 16
    for epoch in range(args.epochs):

        history = model.fit([train_encoder_input, train_decoder_input], train_decoder_output,
              validation_data=([val_encoder_input, val_decoder_input], val_decoder_output),
              batch_size=batch_size,
              epochs=1,
            #   callbacks=[checkpoint]
              )
    # 保存训练集和验证集的loss
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        predicted_sequence = model.predict([test_encoder_input, test_decoder_input],use_multiprocessing = True)
        test_bleu = compute_bleu_score(predicted_sequence, test_decoder_input, output_index_to_word)
        if model_type == "LSTM":
            lstm_bleu_scores.append(test_bleu)
            lstm_train_losses.append(train_loss)
            lstm_val_losses.append(val_loss)
        elif model_type == "GRU":
            gru_bleu_scores.append(test_bleu)
            gru_train_losses.append(train_loss)
            gru_val_losses.append(val_loss)
        else:
            rnn_bleu_scores.append(test_bleu)
            rnn_train_losses.append(train_loss)
            rnn_val_losses.append(val_loss)
        print(f'Test BLEU-4 Score for {model_type} after epoch {epoch+1}: {test_bleu}')

    if model_type == "LSTM":
        plt.plot(range(1, args.epochs + 1), lstm_bleu_scores, marker='o')
        plt.title(f'BLEU Score for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('BLEU Score')
        plt.savefig(f'./image/bleu_score_{model_type}.png')  # 保存BLEU图到image文件夹
        plt.close()  # 关闭图表

        plt.plot(range(1, args.epochs + 1), lstm_train_losses, marker='o', label='Train Loss')
        plt.plot(range(1, args.epochs + 1), lstm_val_losses, marker='o', label='Val Loss')
        plt.title(f'Train and Val Loss for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'image/loss_curve_{model_type}.png')  # 保存loss图到image文件夹
        plt.close()  # 关闭图表
    elif model_type == "GRU":
        plt.plot(range(1, args.epochs + 1), gru_bleu_scores, marker='o')
        plt.title(f'BLEU Score for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('BLEU Score')
        plt.savefig(f'./image/bleu_score_{model_type}.png')  # 保存BLEU图到image文件夹
        plt.close()  # 关闭图表

        plt.plot(range(1, args.epochs + 1), gru_train_losses, marker='o', label='Train Loss')
        plt.plot(range(1, args.epochs + 1), gru_val_losses, marker='o', label='Val Loss')
        plt.title(f'Train and Val Loss for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'image/loss_curve_{model_type}.png')  # 保存loss图到image文件夹
        plt.close()  # 关闭图表
    else:
        plt.plot(range(1, args.epochs + 1), rnn_bleu_scores, marker='o')
        plt.title(f'BLEU Score for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('BLEU Score')
        plt.savefig(f'./image/bleu_score_{model_type}.png')  # 保存BLEU图到image文件夹
        plt.close()  # 关闭图表

        plt.plot(range(1, args.epochs + 1), rnn_train_losses, marker='o', label='Train Loss')
        plt.plot(range(1, args.epochs + 1), rnn_val_losses, marker='o', label='Val Loss')
        plt.title(f'Train and Val Loss for {model_type} Models')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'image/loss_curve_{model_type}.png')  # 保存loss图到image文件夹
        plt.close()  # 关闭图表   
    model_dir = "model_save"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = f"{model_type}.h5"
    model_file = os.path.join(model_dir, model_name)
    model.save(model_file)

plt.plot(range(1, args.epochs + 1), lstm_bleu_scores, marker='o', label='LSTM BLEU Score')
plt.plot(range(1, args.epochs + 1), gru_bleu_scores, marker='o', label='GRU BLEU Score')
plt.plot(range(1, args.epochs + 1), rnn_bleu_scores, marker='o', label='RNN BLEU Score')
plt.title('BLEU Score for all Models')
plt.xlabel('Epochs')
plt.ylabel('Bleu')
plt.legend()
plt.savefig('./image/bleu_score_all.png')  # 保存loss图到image文件夹
plt.close()  # 关闭图表


# # 画出BLEU分数折线图
#     plt.plot(range(1, args.epochs + 1), bleu_scores, marker='o')
#     plt.title(f'BLEU Score for {model_type} Models')
#     plt.xlabel('Epochs')
#     plt.ylabel('BLEU Score')
#     plt.savefig(f'./image/bleu_score_{model_type}.png')  # 保存BLEU图到image文件夹
#     plt.close()  # 关闭图表

# # 画出训练集和验证集的loss折线图
#     plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
#     plt.plot(range(1, args.epochs + 1), val_losses, marker='o', label='Val Loss')
#     plt.title(f'Train and Val Loss for {model_type} Models')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'image/loss_curve_{model_type}.png')  # 保存loss图到image文件夹
#     plt.close()  # 关闭图表

