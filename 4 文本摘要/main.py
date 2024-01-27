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


train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 0.25比例划分训练集和验证集
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# 使用map()函数将split()函数应用到train_data['description']中的每个元素上，计算每个元素拆分后的长度，然后使用max()函数找到最大的长度。
max_input_len = max(train_data['description'].map(lambda x: len(x.split())))
max_output_len = max(train_data['diagnosis'].map(lambda x: len(x.split())))

# 构建词汇表, 长度+1特殊标记
i_vocab = set(' '.join(train_data['description']).split())
i_size = len(i_vocab) + 1

o_vocab = set(' '.join(train_data['diagnosis']).split())
o_size = len(o_vocab) + 1

# 索引号从1开始，而不是从0开始，这是因为在后续模型训练过程中，通常会使用0作为填充值的索引
i_word_to_index = {word: index + 1 for index, word in enumerate(i_vocab)}
i_index_to_word = {index + 1: word for index, word in enumerate(i_vocab)}
o_word_to_index = {word: index + 1 for index, word in enumerate(o_vocab)}
o_index_to_word = {index + 1: word for index, word in enumerate(o_vocab)}


# 填充
train_encoder_input = pad(train_data['description'], i_word_to_index, max_input_len)
train_decoder_input = pad(train_data['diagnosis'], o_word_to_index, max_output_len)
train_decoder_output = pad(train_data['diagnosis'], o_word_to_index, max_output_len)

val_encoder_input = pad(val_data['description'], i_word_to_index, max_input_len)
val_decoder_input = pad(val_data['diagnosis'], o_word_to_index, max_output_len)
val_decoder_output = pad(val_data['diagnosis'], o_word_to_index, max_output_len)

test_encoder_input = pad(test_data['description'], i_word_to_index, max_input_len)
test_decoder_input = pad(test_data['diagnosis'], o_word_to_index, max_output_len)


# 构建Seq2Seq模型
latent_dim = 64

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default='LSTM')

args = parser.parse_args()
if args.model == "LSTM":
    model = LSTM (max_input_len,max_output_len,i_size,o_size,latent_dim)
elif args.model == "GRU":
    model = GRU (max_input_len,max_output_len,i_size,o_size,latent_dim)
else:
    model = RNN (max_input_len,max_output_len,i_size,o_size,latent_dim)

bleu_scores = []  # 用于保存每个epoch的BLEU分数
rouge_scores = []  # 用于保存每个epoch的ROUGE分数
train_losses = [] # 用于保存每个epoch的训练集loss
val_losses = []   # 用于保存每个epoch的验证集loss
# 定义回调函数保存最佳模型
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# 训练模型
batch_size = 4
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
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    predicted_sequence = model.predict([test_encoder_input, test_decoder_input],use_multiprocessing = True)
    test_bleu = compute_bleu_score(predicted_sequence, test_decoder_input, o_index_to_word)
    bleu_scores.append(test_bleu)
    print(f'Test BLEU-4 Score after epoch {epoch+1}: {test_bleu}')

# 画出BLEU分数折线图
plt.plot(range(1, args.epochs + 1), bleu_scores, marker='o')
plt.title('BLEU Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('BLEU Score')
plt.savefig('./image/bleu_score.png')  # 保存BLEU图到image文件夹
plt.close()  # 关闭图表

# 画出训练集和验证集的loss折线图
plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, args.epochs + 1), val_losses, marker='o', label='Val Loss')
plt.title('Train and Val Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('image/loss_curve.png')  # 保存loss图到image文件夹
plt.close()  # 关闭图表

model_dir = "model_save"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_name = f"{args.model}.h5"
model_file = os.path.join(model_dir, model_name)
model.save(model_file)



