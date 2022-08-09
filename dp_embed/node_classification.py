from utils import splitdata
import argparse
import torch
from model.MLP import MLP
import torch.nn.functional as F
import numpy as np
import os
from unsup_align.run_align import make_unsuper_align
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def train(train_x, train_label):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.MultiLabelMarginLoss()  # 定义损失标准
    x = train_x
    label = train_label
    pred = model(x)
    loss = criterion(pred, label)
    loss.backward(retain_graph=True)
    optimizer.step()

    return model, loss


def evaluate(data, label):
    model.eval()
    pred = model(data)
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失标准
    loss = criterion(pred, label)

    pred_label = F.softmax(pred).argmax(axis=1)
    acc = float(((label == pred_label).sum() / label.shape[0]))

    return loss, acc


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


def multi_evaluate(data, label):
    model.eval()
    pred = model(data)
    # print("pred:",pred)
    criterion = torch.nn.MultiLabelMarginLoss()
    loss = criterion(pred, label)

    pred[torch.where(pred >= 0.5)] = 1
    pred[torch.where(pred < 0.5)] = 0
    # print("new pred:",pred)
    acc = Accuracy(pred.detach().numpy(), label.detach().numpy())
    p = precision_score(pred.detach().numpy(), label.detach().numpy(), average="micro")
    f1_mi = f1_score(pred.detach().numpy(), label.detach().numpy(), average="micro")
    f1_ma = f1_score(pred.detach().numpy(), label.detach().numpy(), average="macro")
    return loss, p, acc, f1_mi, f1_ma


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="foursquare")
    parser.add_argument("--assist_dataset", default="twitter")
    parser.add_argument("--labelfile", type=str, default="../dataset/%s/data/user_label2id.pkl")
    parser.add_argument("--embfile", type=str, default="../dataset/%s/data/")
    parser.add_argument("--attrfile", default="")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--emb_dir", )
    args = parser.parse_args()

    if args.dataset == "twitter" or args.dataset == "foursquare":
        # args.labelfile = "../dataset/%s/data/user_label2id.pkl"
        args.labelfile = "../dataset/%s/data/multilabel2id.pkl"
        args.genderlabelfile = "../dataset/%s/data/genderlabel2id.pkl"
        args.occulabelfile = "../dataset/%s/data/userocculabel2id.pkl"
        args.emb_dir = "../dataset/%s/data/"
    elif args.dataset == "weibo1" or args.dataset == "weibo2":
        args.labelfile = "../dataset/weibo/data/%s/user_label{}2id.pkl".format(args.dataset[-1])
        args.emb_dir = "../dataset/weibo/data/%s/"
        args.occulabelfile = "../dataset/weibo/data/%s/user_occu_label{}2id.pkl".format(args.dataset[-1])
        args.genderlabelfile = "../dataset/weibo/data/%s/user_gender_label{}2id.pkl".format(args.dataset[-1])

    if args.mode == 0:
        args.embfile = args.emb_dir % args.dataset + "origin/useremb.pt"
        print("get emb from:", args.embfile)
        emb = torch.load(args.embfile)


    elif args.mode == 1:
        args.embfile = args.emb_dir % args.dataset + "perturbed/perturb_useremb.pt"
        print("get emb from:", args.embfile)
        emb = torch.load(args.embfile)

    elif args.mode == 2:
        args.embfile = args.emb_dir % args.dataset + "/perturbed/joint_perturb_useremb.pt"
        # args.embfile = args.emb_dir % args.dataset + "/origin/joint_useremb.pt"
        print("get emb from:", args.embfile)
        emb = torch.load(args.embfile)

    else:
        raise NotImplementedError

    print(emb.shape)

    train_id, train_label, val_id, val_label = splitdata(args.labelfile % args.dataset, 0.8, 0.2)
    train_x = emb[train_id]
    val_x = emb[val_id]

    p_list = []
    f1_mi_list = []
    for i in range(5):
        cls = DecisionTreeClassifier()
        # train
        # print(train_x.detach().numpy())
        cls.fit(train_x.detach().numpy(), train_label.detach().numpy())
        # predict
        pred = cls.predict(val_x.detach().numpy())
        # print(pred)
        pred[np.where(pred >= 0.5)] = 1
        pred[np.where(pred < 0.5)] = 0
        # print("new pred:",pred)
        # acc = Accuracy(pred, val_label.detach().numpy())
        p = precision_score(pred, val_label.detach().numpy(), average="micro")
        f1_mi = f1_score(pred, val_label.detach().numpy(), average="micro")
        f1_ma = f1_score(pred, val_label.detach().numpy(), average="macro")
        print("p:", p, 'f1_mi:', f1_mi, 'f1_ma:', f1_ma)
        p_list.append(p)
        f1_mi_list.append(f1_mi)



