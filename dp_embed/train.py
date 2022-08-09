from utils import construct_heter_graph_from_file,construct_joint_heter_graph_from_file
import torch
from model.HGNN import HeteroGCN
import argparse
import torch.nn.functional as F
import torch.optim as optim
from utils import get_triplet
import numpy as np


def train(G, fea_dict, fea_dim_dict, usernum, epoch, hid_dim, out_dim, triplet_size, save_path, joint=False, usernumB = 0, save_pathB = None, align_index = None):
    model = HeteroGCN(fea_dim_dict, hid_dim, out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.train()
    if joint:
        indexA = align_index[:,0]
        indexB = align_index[:,1] + np.array([usernum] * indexA.shape[0])
    for e in range(epoch):
        optimizer.zero_grad()
        outemb = model(G, fea_dict)
        triplets = get_triplet(G, usernum+usernumB, triplet_size)
        pos_loss = F.logsigmoid((outemb[triplets[:,0]] * outemb[triplets[:,1]]).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(outemb[triplets[:,0]] * outemb[triplets[:,2]]).sum(-1)).mean()
        if joint:
            dis = (outemb[indexA] - outemb[indexB]).pow(2).sum(1).mean()
            total_loss = -pos_loss - neg_loss + 100*dis
        else:
            total_loss = -pos_loss - neg_loss
        print(e,total_loss)
        total_loss.backward()
        optimizer.step()
    if joint:
        torch.save(outemb[0:usernum,:],save_path)
        torch.save(outemb[usernum:usernum+usernumB,:],save_pathB)
        print("successfully save user embedding in %s and %s" % (save_path,save_pathB))
    else:
        torch.save(outemb, save_path)
        print("successfully save user embedding in %s"%save_path)

def evaluate():
    pass


def set_args(args):

    if args.mode == 0:
        args.followfile = "../dataset/%s/data/origin/following.npy"%args.dataset
        args.textvecfile = "../dataset/%s/data/origin/user_text_vec2id.pkl"%args.dataset
        args.attfile = "../dataset/%s/data/origin/userattr.npy"%args.dataset
        args.save_path = "../dataset/%s/data/origin/useremb.pt"%args.dataset
        if args.dataset == "weibo1" or args.dataset=="weibo2":
            args.followfile = "../dataset/weibo/data/%s/origin/following.npy" % args.dataset
            args.textvecfile = "../dataset/weibo/data/%s/origin/user_text_vec2id.pkl" % args.dataset
            args.attfile = "../dataset/weibo/data/%s/origin/userattr.npy" % args.dataset
            args.save_path = "../dataset/weibo/data/%s/origin/useremb.pt" % args.dataset
    elif args.mode == 1:
        args.followfile = "../dataset/%s/data/perturbed/tmf_eps:5.npy"%args.dataset
        args.textvecfile = "../dataset/%s/data/perturbed/san_user_text_vec_15.00_0.50_0.202id.pkl" % args.dataset
        args.attfile = "../dataset/%s/data/perturbed/userattr-5.npy" % args.dataset
        args.save_path = "../dataset/%s/data/perturbed/perturb_useremb.pt" % args.dataset
        if args.dataset == "weibo1" or args.dataset == "weibo2":
            args.followfile = "../dataset/weibo/data/%s/perturbed/tmf_eps:4.npy" % (args.dataset)
            args.textvecfile = "../dataset/weibo/data/%s/perturbed/san_user_text_vec_15.00_0.50_0.202id.pkl" % args.dataset
            args.attfile = "../dataset/weibo/data/%s/perturbed/userattr-5.npy" % args.dataset
            args.save_path = "../dataset/weibo/data/%s/perturbed/perturb_useremb.pt" % args.dataset
  

    else:
        raise NotImplementedError
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="twitter")
    parser.add_argument("--datasetA", default="foursquare")
    parser.add_argument("--datasetB", default="twitter")
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--triplet_size", type=int, default=50000)

    parser.add_argument("--followfile",type=str, default="")
    parser.add_argument("--textvecfile", type=str, default="")
    parser.add_argument("--attfile", type=str, default="")
    parser.add_argument("--save_path",type = str, default="")
    parser.add_argument("--align_index_file", type=str, default="")
    parser.add_argument("--mode",type=int,default=0)

    args = parser.parse_args()
    args = set_args(args)

    if args.mode == 0 or args.mode == 1:
        G, fea_dict, fea_dim_dict, usernum = construct_heter_graph_from_file(args.followfile, args.textvecfile, args.attfile)
        print(G)
        train(G, fea_dict, fea_dim_dict, usernum, args.epoch, args.hid_dim, args.out_dim, args.triplet_size, args.save_path)



main()