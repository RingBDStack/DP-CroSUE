from utils import construct_heter_graph_from_file, construct_joint_heter_graph_from_file
from model.HGNN import HeteroGCN
import argparse
import torch
import torch.nn.functional as F
from utils import get_triplet
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor

ms.set_context(device_target='CPU', device_id=0)


class LogSigmoid(nn.Cell):

    def __init__(self):
        """Initialize LogSigmoid."""
        super(LogSigmoid, self).__init__()
        self.mul = ops.Mul()
        self.exp = ops.Exp()
        self.add = ops.Add()

    def construct(self, input_x):
        # torch.clamp(x,
        #             max=0) - torch.log(torch.exp(-torch.abs(x)) +
        #                                1) + 0.5 * torch.clamp(x, min=0, max=0)

        neg_input = self.mul(input_x, -1)
        max_negx_zero = ops.maximum(neg_input, Tensor([0], ms.float32))
        return -(max_negx_zero + ops.log(
            self.exp(self.mul(max_negx_zero, -1)) +
            self.exp(self.add(neg_input, self.mul(max_negx_zero, -1)))))
        # return self.mul(ops.log(self.add(1, self.exp(self.mul(input_x, -1)))),
        #                 -1)
        # print('######')
        # print(neg_input.min())
        # print(neg_input.max())
        # exp_neg_input = self.exp(neg_input.astype(ms.float64))
        # #精度问题
        # print(exp_neg_input.max())
        # print(exp_neg_input.min())
        # exp_neg_input_1 = self.add(exp_neg_input, 1)
        # print(exp_neg_input_1.max())
        # print(exp_neg_input_1.min())
        # rec_exp_neg_input_1 = self.rec(exp_neg_input_1.astype(ms.float32))
        # print(rec_exp_neg_input_1.max())
        # print(rec_exp_neg_input_1.min())
        # print('######')
        # # ret = ops.log(
        # #     ops.clip_by_value(rec_exp_neg_input_1, Tensor(1e-32, ms.float32),
        # #                       Tensor(1e32, ms.float32)))
        # ret = ops.log(rec_exp_neg_input_1)
        # return ret


class Loss(nn.LossBase):

    def __init__(self):
        super(Loss, self).__init__()

    def construct(self, outemb, triplets):
        pos_loss = LogSigmoid()(ops.ReduceSum()(
            outemb[triplets[:, 0]] * outemb[triplets[:, 1]], -1)).mean()
        print(pos_loss)
        neg_loss = LogSigmoid()(ops.ReduceSum()(
            -(outemb[triplets[:, 0]] * outemb[triplets[:, 2]]), -1)).mean()
        print(neg_loss)
        return -pos_loss - neg_loss


outemb = torch.load('./outemb')
triplet = torch.load('./triplets')
pos_loss = F.logsigmoid(
    (outemb[triplet[:, 0]] * outemb[triplet[:, 1]]).sum(-1)).mean()
neg_loss = F.logsigmoid(-(outemb[triplet[:, 0]] *
                          outemb[triplet[:, 2]]).sum(-1)).mean()
print(f'torch pos_loss: {pos_loss}')
print(f'torch neg_loss: {neg_loss}')
a = np.array([[0, 1, 2], [3, 4, 5]])
outemb = Tensor(outemb.detach().numpy(), ms.float32)
triplet = Tensor(triplet.numpy(), ms.int32)
loss_fn = Loss()
loss_fn(outemb, triplet)
print(LogSigmoid()(Tensor([-300.0], ms.float32)))