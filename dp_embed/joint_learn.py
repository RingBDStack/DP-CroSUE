from utils import construct_heter_graph_from_file
import torch
from model.joint_convolution import joint_GCN, share_joint_GCN
import argparse
import torch.nn.functional as F
import torch.optim as optim
from utils import get_triplet
from unsup_align.run_align import make_unsuper_align
import os
import numpy as np
from scipy import sparse
from utils import normalize
from utils import get_multi_adj
from utils import get_triplet_from_edge_file
import dgl


def train(embA, embB, multi_adjA, multi_adjB, align_index, nodesA, nodesB, args):
    modelA = joint_GCN(args.in_dim, args.out_dim)
    modelB = joint_GCN(args.in_dim, args.out_dim)
    optimizerA = optim.Adam(modelA.parameters(), lr=0.003, weight_decay=1e-4)
    optimizerB = optim.Adam(modelA.parameters(), lr=0.003, weight_decay=1e-4)

    for e in range(args.epoch):
        modelA.train()
        modelB.train()
        optimizerA.zero_grad()
        optimizerB.zero_grad()
        outembA = modelA(multi_adjA, embA, embB, align_index[:, 0], align_index[:, 1])
        outembB = modelB(multi_adjB, embB, embA, align_index[:, 1], align_index[:, 0])
        tripletsA = get_triplet_from_edge_file(args.followfileA, nodesA, args.triplet_size)
        tripletsB = get_triplet_from_edge_file(args.followfileB, nodesB, args.triplet_size)
        pos_lossA = F.logsigmoid((outembA[tripletsA[:, 0]] * outembA[tripletsA[:, 1]]).sum(-1)).mean()
        neg_lossA = F.logsigmoid(-(outembA[tripletsA[:, 0]] * outembA[tripletsA[:, 2]]).sum(-1)).mean()
        pos_lossB = F.logsigmoid((outembB[tripletsB[:, 0]] * outembB[tripletsB[:, 1]]).sum(-1)).mean()
        neg_lossB = F.logsigmoid(-(outembB[tripletsB[:, 0]] * outembB[tripletsB[:, 2]]).sum(-1)).mean()
        # dis = (outembA[align_index[:, 0]] - outembB[align_index[:, 1]]).pow(2).sum(1).mean()
        l = torch.nn.L1Loss(size_average=True, reduce=True, reduction='average')
        dis = l(outembA[align_index[:, 0]], outembB[align_index[:, 1]])
        total_loss = -pos_lossA - neg_lossA - pos_lossB - neg_lossB + 10 * dis
        print(e, total_loss)
        total_loss.backward()
        optimizerA.step()
        optimizerB.step()
    torch.save(outembA, args.save_pathA)
    torch.save(outembB, args.save_pathB)
    print("successfully save joint user embedding in %s and %s" % (args.save_pathA, args.save_pathB))


def share_W_train(embA, embB, multi_adjA, multi_adjB, align_index, nodesA, nodesB, args):
    model = share_joint_GCN(args.in_dim, args.out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for e in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        outembA, outembB = model(multi_adjA, embA, multi_adjB, embB, align_index[:, 0], align_index[:, 1])
        tripletsA = get_triplet_from_edge_file(args.followfileA, nodesA, args.triplet_size)
        tripletsB = get_triplet_from_edge_file(args.followfileB, nodesB, args.triplet_size)
        pos_lossA = F.logsigmoid((outembA[tripletsA[:, 0]] * outembA[tripletsA[:, 1]]).sum(-1)).mean()
        neg_lossA = F.logsigmoid(-(outembA[tripletsA[:, 0]] * outembA[tripletsA[:, 2]]).sum(-1)).mean()
        pos_lossB = F.logsigmoid((outembB[tripletsB[:, 0]] * outembB[tripletsB[:, 1]]).sum(-1)).mean()
        neg_lossB = F.logsigmoid(-(outembB[tripletsB[:, 0]] * outembB[tripletsB[:, 2]]).sum(-1)).mean()
        # dis = (outembA[align_index[:, 0]] - outembB[align_index[:, 1]]).pow(2).sum(1).mean()
        l = torch.nn.L1Loss(size_average=True, reduce=True, reduction='average')
        dis = l(outembA[align_index[:, 0]], outembB[align_index[:, 1]])
        total_loss = -pos_lossA - neg_lossA - pos_lossB - neg_lossB + 0.1 * dis
        print(e, total_loss)
        total_loss.backward()
        optimizer.step()
    torch.save(outembA, args.save_pathA)
    torch.save(outembB, args.save_pathB)
    print("successfully save joint user embedding in %s and %s" % (args.save_pathA, args.save_pathB))


def evaluate():
    pass


def set_args(args):
    if args.mode == 0:
        if args.datasetA in ["foursquare", "twitter"] and args.datasetB in ["foursquare", "twitter"]:
            args.expfolderA = "../dataset/%s/data/origin/" % args.datasetA
            args.expfolderB = "../dataset/%s/data/origin/" % args.datasetB
        elif args.datasetA in ["weibo1", "weibo2"] and args.datasetB in ["weibo1", "weibo2"]:
            args.expfolderA = "../dataset/weibo/data/%s/origin/" % args.datasetA
            args.expfolderB = "../dataset/weibo/data/%s/origin/" % args.datasetB
        args.followfileA = args.expfolderA + "following.npy"
        args.save_pathA = args.expfolderA + "joint_useremb.pt"
        args.embfileA = args.expfolderA + "useremb.pt"
        args.followfileB = args.expfolderB + "following.npy"
        args.save_pathB = args.expfolderB + "joint_useremb.pt"
        args.embfileB = args.expfolderB + "useremb.pt"

    if args.mode == 1:
        if args.datasetA in ["foursquare", "twitter"] and args.datasetB in ["foursquare", "twitter"]:
            args.expfolderA = "../dataset/%s/data/perturbed/" % args.datasetA
            args.expfolderB = "../dataset/%s/data/perturbed/" % args.datasetB
        elif args.datasetA in ["weibo1", "weibo2"] and args.datasetB in ["weibo1", "weibo2"]:
            args.expfolderA = "../dataset/weibo/data/%s/perturbed/" % args.datasetA
            args.expfolderB = "../dataset/weibo/data/%s/perturbed/" % args.datasetB
        args.followfileA = args.expfolderA + "tmf_eps:5.npy"
        args.followfileB = args.expfolderB + "tmf_eps:5.npy"
        args.embfileA = args.expfolderA + "perturb_useremb.pt"
        args.embfileB = args.expfolderB + "perturb_useremb.pt"
        args.save_pathA = args.expfolderA + "joint_perturb_useremb.pt"
        args.save_pathB = args.expfolderB + "joint_perturb_useremb.pt"

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetA", default="weibo1")
    parser.add_argument("--datasetB", default="weibo2")
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--in_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--triplet_size", type=int, default=50000)

    parser.add_argument("--followfileA", type=str, default="")
    parser.add_argument("--followfileB", type=str, default="")
    parser.add_argument("--embfileA", type=str, default="")
    parser.add_argument("--embfileB", type=str, default="")
    parser.add_argument("--save_pathA", type=str, default="")
    parser.add_argument("--save_pathB", type=str, default="")
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    args = set_args(args)

    print("get embA from:", args.embfileA)
    print("get embB from:", args.embfileB)
    embA = torch.load(args.embfileA)
    embB = torch.load(args.embfileB)
    if args.mode == 0:
        mode = "origin"
    elif args.mode == 1:
        mode = "perturbed"
    align_index_file = "../dataset/align/%s/%s/unsuper_files/%s_%s.npy" % (
        mode, args.datasetA, args.datasetA, args.datasetB)
    if not os.path.exists(align_index_file):
        print("generate original alignment--------")
        make_unsuper_align(args.datasetA, args.datasetB, args.embfileA, args.embfileB,
                           "../dataset/align/%s/%s/" % (mode,args.datasetA))
    align_index = np.load(align_index_file)

    nodesA = embA.shape[0]
    nodesB = embB.shape[0]
    edgesA = np.load(args.followfileA)
    edgesB = np.load(args.followfileB)
    adjA = dgl.DGLGraph((edgesA[:, 0], edgesA[:, 1])).adjacency_matrix().to_dense() + torch.tensor(
        sparse.eye(nodesA).todense())
    adjB = dgl.DGLGraph((edgesB[:, 0], edgesB[:, 1])).adjacency_matrix().to_dense() + torch.tensor(
        sparse.eye(nodesB).todense())

    # adjA = sparse.csr_matrix(np.load(args.followfileA)) + sparse.eye(nodesA).todense()
    # adjB = sparse.csr_matrix(np.load(args.followfileB)) + sparse.eye(nodesB).todense()
    print("generate multi adj-------")
    multi_adjA = get_multi_adj(adjA, nodesA)
    multi_adjB = get_multi_adj(adjB, nodesB)

    train(embA, embB, multi_adjA, multi_adjB, align_index, nodesA, nodesB, args)
    # share_W_train(embA,embB,multi_adjA,multi_adjB,align_index,nodesA,nodesB,args)


main()