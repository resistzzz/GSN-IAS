import argparse
import csv
import pickle
import datetime

# SR-GNN代码处理生成的all_train_seq.txt与test.txt文件中物品数目对应不上
# 该文件则是通过train.txt生成all_train_seq.txt
# train.txt中物品总数为17376，而train.txt和test.txt中物品总数为17745，即有369个物品只出现在测试集中，未出现在训练集中
# 本文件将只出现在测试集中，未出现在训练集中的物品予以剔除


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose')
opt = parser.parse_args()
print(opt)

train_data = pickle.load(open('yoochoose1_64/train.txt', 'rb'))
test_data = pickle.load(open('yoochoose1_64/test.txt', 'rb'))

all_seqs_train = []
for seq, lab in zip(train_data[0], train_data[1]):
    all_seqs_train.append(seq + [lab])
items_train = []
for seq in all_seqs_train:
    for s in seq:
        if s not in items_train:
            items_train.append(s)

# 忽视没有出现在训练集中的物品
out_test_data = [[], []]
for i in range(len(test_data[0])):
    seq, lab = test_data[0][i], test_data[1][i]
    seq = seq + [lab]
    out = []
    for s in seq:
        if s in items_train:
            out.append(s)
    if len(out) >= 2:
        out_test_data[0].append(out[0:-1])
        out_test_data[1].append(out[-1])

# items_test = []
# for seq, lab in zip(out_test_data[0], out_test_data[1]):
#     seq = seq + [lab]
#     for s in seq:
#         if s not in items_test:
#             items_test.append(s)

out_all_seqs = []
cnt = 0
while True:
    if cnt >= len(all_seqs_train):
        break
    out_all_seqs.append(all_seqs_train[cnt])
    cnt += len(all_seqs_train[cnt]) - 1

pickle.dump(out_all_seqs, open('yoochoose1_64/all_train_seq.txt', 'wb'))
pickle.dump(out_test_data, open('yoochoose1_64/test.txt', 'wb'))
print('Done!')




