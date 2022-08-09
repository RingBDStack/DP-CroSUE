import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
import torch.nn.functional as F
class joint_GCN(nn.Module):
    def __init__(self,in_dim,out_dim,bias=True):
        super(joint_GCN,self).__init__()
        self.inter_weight =  Parameter(torch.FloatTensor(in_dim,out_dim))
        self.intra_weight = Parameter(torch.FloatTensor(in_dim,out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.drop_layer = nn.Dropout(0.2)


    def reset_parameters(self):
        inter_stdv = 1. / math.sqrt(self.inter_weight.size(1))
        intra_stdv = 1. / math.sqrt(self.intra_weight.size(1))
        self.inter_weight.data.uniform_(-inter_stdv, inter_stdv)
        self.intra_weight.data.uniform_(-intra_stdv, intra_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-intra_stdv, intra_stdv)

    def forward(self,adj,X,assist_X,align_indexA,align_indexB):
        X[align_indexA].add_(torch.mm(assist_X[align_indexB],self.inter_weight))
        X[align_indexA].div_(torch.FloatTensor((2,)))

        suport = torch.mm(X,self.intra_weight)
        #print(type(suport),suport)
        out_emb = torch.mm(torch.FloatTensor(adj),suport)

        if self.bias is not None:
            return out_emb + self.bias
        else:
            return out_emb


class GCN(nn.Module):
    def __init__(self,in_dim,out_dim,bias=True):
        super(joint_GCN,self).__init__()
        self.linear1 = nn.Linear(in_dim,out_dim)
        self.linear2 = nn.Linear(in_dim,out_dim)
        self.linear3 = nn.Linear(in_dim,out_dim)
        self.inter_weight = Parameter(torch.FloatTensor(in_dim, out_dim))

    def reset_parameters(self):
        inter_stdv = 1. / math.sqrt(self.inter_weight.size(1))
        self.inter_weight.data.uniform_(-inter_stdv, inter_stdv)

    def forward(self,adj,X,assist_X,align_indexA,align_indexB):
        X[align_indexA].add_(torch.mm(assist_X[align_indexB],self.inter_weight))
        X[align_indexA].div_(torch.FloatTensor((2,)))

        suport = self.linear1(X)
        hid1 = torch.mm(torch.FloatTensor(adj),suport)
        hid1 = torch.relu(hid1)
        hid2 = torch.mm(torch.FloatTensor(adj),self.linear2(hid1))
        hid2 = torch.relu(hid2)
        out = torch.mm(torch.FloatTensor(adj),self.linear2(hid2))

        return out



class share_joint_GCN(nn.Module):
    def __init__(self,in_dim,out_dim,bias=True):
        super(share_joint_GCN,self).__init__()
        self.inter_weight =  Parameter(torch.FloatTensor(in_dim,out_dim))
        self.intra_weight = Parameter(torch.FloatTensor(in_dim,out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.drop_layer = nn.Dropout(0.2)


    def reset_parameters(self):
        inter_stdv = 1. / math.sqrt(self.inter_weight.size(1))
        intra_stdv = 1. / math.sqrt(self.intra_weight.size(1))
        self.inter_weight.data.uniform_(-inter_stdv, inter_stdv)
        self.intra_weight.data.uniform_(-intra_stdv, intra_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-intra_stdv, intra_stdv)

    def forward(self,adj,X,assist_adj,assist_X,align_indexA,align_indexB):
        suport = torch.mm(X,self.intra_weight)
        out_emb = torch.mm(torch.FloatTensor(adj),suport)
        assist_suport = torch.mm(assist_X, self.intra_weight)
        assist_out_emb = torch.mm(torch.FloatTensor(assist_adj), assist_suport)
        out_emb = F.relu(self.drop_layer(out_emb))
        assist_out_emb = F.relu(self.drop_layer(assist_out_emb))
        if self.bias is not None:
            return out_emb + self.bias,assist_out_emb+self.bias
        else:
            return out_emb, assist_out_emb

