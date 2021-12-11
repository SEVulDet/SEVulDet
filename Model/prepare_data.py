from sklearn.ensemble import RandomForestClassifier as RFC
from vectorize_gadget import GadgetVectorizer
from clean_gadget import clean_gadget
from sklearn.externals import joblib
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import *
from cnn import CnnNet
import numpy as np
import torch

# hyparameters
seed = 1
lr = 0.0001
epoches = 20
batch_size = 16
output_size = 1

hidden_size = 256
num_layers = 2
embed_dim = 30
bidirectional = True
dropout = 0.5
use_cuda = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 30 in line and gadget:
                yield clean_gadget(gadget), gadget_val
                gadget = []
            elif stripped.split()[0].isdigit():
                if gadget:
                    if stripped.isdigit():
                        gadget_val = int(stripped)
                    else:
                        gadget.append(stripped)
            else:
                gadget.append(stripped)


def get_vectors(filename, vector_length=100):
    gadgets, count = [], 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vectorizer.add_gadget(gadget)
        row = {"gadget": gadget, "val": val, "index": count}
        gadgets.append(row)
    print("Training model...", end="\r")
    vectorizer.train_model(filename)
    print()
    vectors, count = [], 0
    for gadget in gadgets:
        count += 1
        print("Processing gadgets...", count, end="\r")
        vector = vectorizer.vectorize(gadget["gadget"])
        row = {"vector": vector, "val": gadget["val"], "index": gadget["index"]}
        vectors.append(row)
    print()
    return vectors


def getDataLoader(filename):
    parse_file(filename)
    vector_length = 30
    gadgets_vectors = get_vectors(filename, vector_length)
    x, y, index = [], [], []
    loop = 0
    while True:
        if len(gadgets_vectors) == 0:
            break
        gadgets_vector = gadgets_vectors.pop(0)
        x.append(gadgets_vector['vector'].tolist())
        y.append(gadgets_vector['val'])
        index.append(gadgets_vector['index'])
        del gadgets_vector
        loop += 1
        if loop % 10000 == 0:
            print("getDataLoaderPoint", len(gadgets_vectors))
    del gadgets_vectors
    return np.array(x).tolist(), np.array(y).tolist(), np.array(index).tolist()


def log(log):
    with open(save_filename + '.txt', 'a+') as f:
        f.write(log + '\n')
        f.close()
    print(log)


def evaluation_metrics(trues, preds):
    m = confusion_matrix(trues, preds, labels=[0, 1])
    print(m)
    TN, FP, FN, TP = m[0][0], m[0][1], m[1][0], m[1][1]
    A = float((TP + TN) / (TP + FP + TN + FN))  
    R = float((TP) / (TP + FN)) 
    S = float((TN) / (TN + FP))  
    P = float((TP) / (TP + FP))  
    FPR = float(FP / (FP + TN))
    FNR = float(FN / (FN + TP))
    F1 = float((2 * P * (1 - FNR)) / (P + 1 - FNR))
    return A, R, S, P, FPR, FNR, F1


def evaluate(last=False):
    cnn_net.eval()
    trues, preds = [], []
    for calc, (data, label, index) in enumerate(test_loader):
        data = Variable(data).cuda().float().reshape(1, -1, embed_dim)
        pred = cnn_net(data)
        pred = torch.sigmoid(pred)
        trues.append(label.cpu().data.numpy().tolist())
        preds += [1 if i[0] >= 0.85 else 0 for i in pred.cpu().data.numpy().tolist()]
    A, R, S, P, FPR, FNR, F1 = evaluation_metrics(trues, preds)
    if last:
        log(
            'FNR:{:2.4f}, FPR:{:2.4f}, Accuracy:{:2.4f}, Rec_P:{:2.4f}, Rec_N:{:2.4f}, Precision:{:2.4f}, F1:{:2.4f}'.format(
                FNR, FPR, A, R, S, P, F1))
    else:
        log(
            'Epoch:{}, Step:{}, Aver_loss:{:2.8f}, Cur_loss:{:2.8f}, FNR:{:2.4f}, FPR:{:2.4f}, Accuracy:{:2.4f}, Rec_P:{:2.4f},Rec_N:{:2.4f}, Precision:{:2.4f}, F1:{:2.4f}'.format(
                epoch, step, total_loss / total_step, cur_loss.item(), FNR, FPR, A, R, S, P, F1))
    cnn_net.train()


class MyDataset(Data.Dataset):
    def __init__(self, gadgets, labels, indexs):
        self.gadgets = gadgets
        self.labels = labels
        self.indexs = indexs

    def __getitem__(self, index):
        return self.gadgets[index], self.labels[index], self.indexs[index]

    def __len__(self):
        return len(self.gadgets)


def collate_fn(batch):
    for item in batch:
        x, y, ind = item[0], item[1], item[2]
        x, y = np.array(x), np.array(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if use_cuda:
            x, y = x.float().cuda(), y.cuda()
    return [x.view(1, -1, 30), y, ind]
