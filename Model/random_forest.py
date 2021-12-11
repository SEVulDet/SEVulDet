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
import os


def evaluate_with_RF(cnn_net, train_loader, test_loader):
    # hyperparameters
    max_depth = 16
    cnn_net.eval()
    dense_output_test = dense_output_train = []
    truths_test = truths_train = []
    for step, (slice, label) in enumerate(test_loader):
        cnn_net(Variable(slice.reshape(batch_size, 1, sequence_length, embed_dim)).cuda().float())
        dense_output_test += cnn_net.dense_output.cpu().data.numpy().tolist()
        truths_test.append(label.cpu().data.numpy().tolist())

    for step, (slice, label) in enumerate(train_loader):
        cnn_net(Variable(slice.reshape(batch_size, 1, sequence_length, embed_dim)).cuda().float())
        dense_output_train += cnn_net.dense_output.cpu().data.numpy().tolist()
        truths_train.append(label.cpu().data.numpy().tolist())
    clf = RFC(max_depth=max_depth)
    clf.fit(dense_output_train, truths_train)
    preds_RF_test = clf.predict(dense_output_test)
    m = confusion_matrix(truths_test, preds_RF_test, labels=[0, 1])
    FPR = float(m[1][0] / (m[1][0] + m[0][0]))
    FNR = float(m[0][1] / (m[0][1] + m[1][1]))
    A, R, P, F1 = accuracy_score(truths_test, preds_RF_test), recall_score(truths_test,
                                                                           preds_RF_test), precision_score(
        truths_test, preds_RF_test), f1_score(truths_test, preds_RF_test)
    log('RF: FNR:{:2.4f}, FPR:{:2.4f}, Accuracy:{:2.4f}, Recall:{:2.4f}, Precision:{:2.4f}, F1:{:2.4f}'.format(FNR, FPR,
                                                                                                               A, R, P, F1))
    joblib.dump(clf, 'random_forest.pkl')

