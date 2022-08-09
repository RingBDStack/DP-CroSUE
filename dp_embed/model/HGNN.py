import dgl.function as fn
import torch.nn as nn
import torch
import torch.nn.functional as F

class HeteroGCNLayer(nn.Module):
    def __init__(self, in_size_dict, out_size, use_residual = True):
        super(HeteroGCNLayer, self).__init__()
        # W for each nodetype
        self.weight = nn.ModuleDict({
            name : nn.Linear(in_size_dict[name], out_size) for name in in_size_dict
        })
        self.use_residual = use_residual

    def forward(self, G, fea_dict):
        # input输入是每一种节点特征的字典
        for ntype in G.ntypes:
            # 计算 W_r * h
            Wh = self.weight[ntype](fea_dict[ntype].to(torch.float32))
            G.nodes[ntype].data['hid'] = Wh

        funcs = {}
        for etype in G.etypes:
            funcs[etype] = (fn.copy_u('hid', 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        if self.use_residual:
            return {ntype : G.nodes[ntype].data['h'] + G.nodes[ntype].data['hid'] for ntype in G.ntypes}
        else:
            return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}





class HeteroGCN(nn.Module):
    def __init__(self,in_size_dict, hidden_size, out_size):
        super(HeteroGCN, self).__init__()
        # 创建神经网络层
        self.layer1 = HeteroGCNLayer(in_size_dict, hidden_size)
        self.layer2 = HeteroGCNLayer(dict(zip(in_size_dict.keys(),[hidden_size]*len(in_size_dict))), out_size)

    def forward(self, G, fea_dict, joint=False):
        h_dict = self.layer1(G, fea_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        #获取用户节点表示
        if joint:
            return h_dict['userA'], h_dict['userB']
        else:
            return h_dict['user']
