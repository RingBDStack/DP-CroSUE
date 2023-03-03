import dgl.function as fn

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.parameter import Parameter

import numpy as np


class HeteroGCNLayer(nn.Cell):

    def __init__(self, in_size_dict, out_size, use_residual=True):
        super(HeteroGCNLayer, self).__init__()
        # W for each nodetype
        self.weight = nn.CellList(
            [nn.Dense(in_size_dict[name], out_size) for name in in_size_dict])
        self.use_residual = use_residual
        self.out_size = out_size

    def construct(self, node_dict, edge_dict, fea_dict):
        # input输入是每一种节点特征的字典
        # G: {'ntype1', 'ntype2'}
        # fee_dict: {'ntype1': Tensor, 'ntype2': Tensor}
        # node_dict: {'ntype', Tensor[nid]}
        # edge_dict: {'etype':{ 'edges':Tensor[[src], [dst]]}}
        hidden = {}
        h = {}
        nodeType_list = list(node_dict.keys())
        edgeType_list = list(edge_dict.keys())
        for i, ntype in enumerate(nodeType_list):
            Wh = self.weight[i](fea_dict[ntype])
            hidden[ntype] = Wh
            h[ntype] = ops.Zeros()((node_dict[ntype].shape[0], Wh.shape[-1]),
                                   mindspore.float32)
        src = {
            'are posted by': 'texts',
            'self related': 'texts',
            'follow': 'user',
            'hasfan': 'user'
        }
        dst = {
            'are posted by': 'user',
            'self related': 'texts',
            'follow': 'user',
            'hasfan': 'user'
        }
        for i, etype in enumerate(edgeType_list):
            srctype = src[etype]
            dsttype = dst[etype]
            edges = edge_dict[etype]['edges']
            e_m = hidden[srctype][edges[0]]
            mailbox = ops.tensor_scatter_add(
                ops.Zeros()((node_dict[dsttype].shape[0], self.out_size),
                            mindspore.float32),
                edges[1].astype(mindspore.int32).reshape(-1, 1), e_m)
            cnt = ops.tensor_scatter_add(
                ops.Zeros()((node_dict[dsttype].shape[0], self.out_size),
                            mindspore.int32),
                edges[1].astype(mindspore.int32).reshape(-1, 1),
                ops.Ones()(e_m.shape, mindspore.int32))
            mailbox = mailbox / ops.maximum(cnt, Tensor(1, mindspore.int32))
            h[dsttype] += mailbox
        if self.use_residual:
            for ntype in nodeType_list:
                h[ntype] += hidden[ntype]
        return h

        # funcs = {}
        # for etype in G.etypes:
        #     funcs[etype] = (fn.copy_u('hid', 'm'), fn.mean('m', 'h'))
        # G.multi_update_all(funcs, 'sum')
        # if self.use_residual:
        #     return {
        #         ntype: G.nodes[ntype].data['h'] + G.nodes[ntype].data['hid']
        #         for ntype in G.ntypes
        #     }
        # else:
        #     return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroGCN(nn.Cell):

    def __init__(self, in_size_dict, hidden_size, out_size):
        super(HeteroGCN, self).__init__()
        # 创建神经网络层
        self.layer1 = HeteroGCNLayer(in_size_dict, hidden_size)
        self.layer2 = HeteroGCNLayer(
            dict(zip(in_size_dict.keys(), [hidden_size] * len(in_size_dict))),
            out_size)

    def construct(self, node_dict, edge_dict, fea_dict, joint=False):
        h_dict = self.layer1(node_dict, edge_dict, fea_dict)
        for e in h_dict.keys():
            h_dict[e] = nn.LeakyReLU()(h_dict[e])
        # h_dict = {k: nn.LeakyReLU()(h) for k, h in h_dict.items()}
        h_dict = self.layer2(node_dict, edge_dict, h_dict)
        #获取用户节点表示
        if joint:
            return h_dict['userA'], h_dict['userB']
        else:
            return h_dict['user']


if __name__ == '__main__':
    model = HeteroGCN({'texts': 300, 'user': 35}, 256, 128)
    print(model.trainable_params())