from utils import construct_heter_graph_from_file, construct_joint_heter_graph_from_file
# import torch
from model.HGNN import HeteroGCN
import argparse
# import torch.nn.functional as F
# import torch.optim as optim
from utils import get_triplet
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor

ms.set_context(device_target='GPU', device_id=0)
# ms.set_context(device_target='CPU')


class LogSigmoid(nn.Cell):
    '''
    使用log_sum_exp逼近， 避免溢出
    '''

    def __init__(self):
        """Initialize LogSigmoid."""
        super(LogSigmoid, self).__init__()
        self.mul = ops.Mul()
        self.exp = ops.Exp()
        self.add = ops.Add()

    def construct(self, input_x):
        neg_input = self.mul(input_x, -1)
        max_x_zero = ops.maximum(neg_input, Tensor([0], ms.float32))
        return -(max_x_zero + ops.log(
            self.exp(self.mul(max_x_zero, -1)) +
            self.exp(self.add(neg_input, self.mul(max_x_zero, -1)))))
        # neg_input = self.mul(input_x, -1)
        # exp_neg_input = self.exp(neg_input)
        # exp_neg_input_1 = self.add(exp_neg_input, 1)
        # rec_exp_neg_input_1 = self.rec(exp_neg_input_1)
        # # rec_exp_neg_input_1 = 1 / exp_neg_input_1
        # ret = ops.log(
        #     ops.clip_by_value(rec_exp_neg_input_1, Tensor(1e-38, ms.float32),
        #                       Tensor(1e32, ms.float32)))
        # # ret = ops.log(rec_exp_neg_input_1)
        # return ret


class Loss(nn.LossBase):

    def __init__(self):
        super(Loss, self).__init__()

    def construct(self, outemb, triplets):
        pos_loss = LogSigmoid()(ops.ReduceSum()(
            outemb[triplets[:, 0]] * outemb[triplets[:, 1]], -1)).mean()
        neg_loss = LogSigmoid()(-ops.ReduceSum()(
            outemb[triplets[:, 0]] * outemb[triplets[:, 2]], -1)).mean()
        return -pos_loss - neg_loss


def train(G,
          fea_dict,
          fea_dim_dict,
          usernum,
          epoch,
          hid_dim,
          out_dim,
          triplet_size,
          save_path,
          joint=False,
          usernumB=0,
          save_pathB=None,
          align_index=None):
    print(fea_dim_dict, hid_dim, out_dim)
    nodeType_list = G.ntypes
    edgeType_list = G.etypes
    node_dict = {
        e: Tensor(G.nodes(e).numpy(), ms.int32)
        for e in nodeType_list
    }
    node_num_dict = {e: G.nodes(e).shape[0] for e in nodeType_list}
    edge_dict = {}
    for e in edgeType_list:
        edge_dict[e] = {
            'edges':
            Tensor([G.edges(etype=e)[0].numpy(),
                    G.edges(etype=e)[1].numpy()], ms.int32)
        }
    for e in fea_dict.keys():
        fea_dict[e] = Tensor(fea_dict[e].numpy(), ms.float32)
    model = HeteroGCN(fea_dim_dict, hid_dim, out_dim)
    optimizer = nn.Adam(model.trainable_params(),
                        learning_rate=1e-3,
                        weight_decay=1e-4)
    loss_fn = Loss()

    def forward_fn(node_dict, edge_dict, fea_dict, triplets):
        outemb = model(node_dict, edge_dict, fea_dict)
        loss = loss_fn(outemb, triplets)
        return loss, outemb

    grad_fn = ops.value_and_grad(forward_fn,
                                 None,
                                 optimizer.parameters,
                                 has_aux=True)

    def train_step(node_dict, edge_dict, fea_dict, triplets):
        (loss, _), grads = grad_fn(node_dict, edge_dict, fea_dict, triplets)
        loss = ops.depend(loss, optimizer(grads))
        return loss, _

    model.set_train()
    if joint:
        indexA = align_index[:, 0]
        indexB = align_index[:, 1] + np.array([usernum] * indexA.shape[0])
    for e in range(epoch):
        triplets = get_triplet(G, usernum + usernumB, triplet_size)
        triplets = Tensor(triplets.numpy(), ms.int32)
        loss, outemb = train_step(node_dict, edge_dict, fea_dict, triplets)
        print(f'Epoch: {e}/{epoch}, loss: {loss}')
    ms.save_checkpoint(model, save_path)
    print("successfully save user_embedding net in %s" % (save_path))
    # if joint:
    #     ms.save_checkpoint(outemb[0:usernum, :], save_path)
    #     ms.save_checkpoint(outemb[usernum:usernum + usernumB, :], save_pathB)
    #     print("successfully save user embedding in %s and %s" %
    #           (save_path, save_pathB))
    # else:
    #     ms.save_checkpoint(outemb, save_path)
    #     print("successfully save user embedding in %s" %
    #           (save_path[:-2] + 'ckpt'))


def evaluate():
    pass


def set_args(args):

    if args.mode == 0:
        args.followfile = "../dataset/%s/data/origin/following.npy" % args.dataset
        args.textvecfile = "../dataset/%s/data/origin/user_text_vec2id.pkl" % args.dataset
        args.attfile = "../dataset/%s/data/origin/userattr.npy" % args.dataset
        args.save_path = "../dataset/%s/data/origin/useremb.ckpt" % args.dataset
        if args.dataset == "weibo1" or args.dataset == "weibo2":
            args.followfile = "../dataset/weibo/data/%s/origin/following.npy" % args.dataset
            args.textvecfile = "../dataset/weibo/data/%s/origin/user_text_vec2id.pkl" % args.dataset
            args.attfile = "../dataset/weibo/data/%s/origin/userattr.npy" % args.dataset
            args.save_path = "../dataset/weibo/data/%s/origin/useremb.pt" % args.dataset
    else:
        raise NotImplementedError
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="foursquare")
    parser.add_argument("--datasetA", default="foursquare")
    parser.add_argument("--datasetB", default="twitter")
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--triplet_size", type=int, default=50000)

    parser.add_argument("--followfile", type=str, default="")
    parser.add_argument("--textvecfile", type=str, default="")
    parser.add_argument("--attfile", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--align_index_file", type=str, default="")
    parser.add_argument("--mode", type=int, default=0)

    args = parser.parse_args()
    args = set_args(args)

    if args.mode == 0:
        G, fea_dict, fea_dim_dict, usernum = construct_heter_graph_from_file(
            args.followfile, args.textvecfile, args.attfile)
        print(G)
        train(G, fea_dict, fea_dim_dict, usernum, args.epoch, args.hid_dim,
              args.out_dim, args.triplet_size, args.save_path)


main()