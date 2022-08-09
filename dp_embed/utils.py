import pickle
import dgl
import numpy as np
import torch
from scipy import sparse
import random

def construct_heter_graph_from_file(userfollowfile,usertextvecfile,userattrfile):
    adj = np.load(userfollowfile)
    src = adj[:,0]
    dst = adj[:,1]



    with open(usertextvecfile,'rb') as f:
        usertextvec = pickle.load(f)
    usertextvec = sorted(usertextvec.items(),key=lambda item:item[0])
    usertextid = [uservec[0] for uservec in usertextvec]
    print(type(usertextid))
    usertextfea = [uservec[1] for uservec in usertextvec]
    G = dgl.heterograph({('user','hasfan','user'):(torch.tensor(dst),torch.tensor(src)),
                         ('user', 'follow', 'user'): (torch.tensor(src), torch.tensor(dst)),
                         #('user','post','texts'):(torch.tensor(usertextid,dtype=torch.int32),
                         #                         torch.tensor(np.arange(len(usertextvec)),dtype=torch.int32)),
                         ('texts', 'self related', 'texts'): (torch.tensor(np.arange(len(usertextvec)), dtype=torch.int32),
                                                     torch.tensor(np.arange(len(usertextvec)), dtype=torch.int32)),
                         ('texts', 'are posted by', 'user'): (
                                                     torch.tensor(np.arange(len(usertextvec)), dtype=torch.int32),
                                                     torch.tensor(usertextid, dtype=torch.int32))
                         })

    if ".pt" in userattrfile:
        userfea = torch.load(userattrfile)
    else:
        userfea = torch.tensor(np.load(userattrfile))
    G.nodes['user'].data['fea'] = userfea
    G.nodes['texts'].data['fea'] = torch.tensor(usertextfea)

    fea_dict = dict()
    fea_dim_dict = dict()
    for ntype in G.ntypes:
        fea = G.nodes[ntype].data['fea']
        fea_dict[ntype] = fea
        fea_dim_dict[ntype] = fea.shape[1]
    usernum = G.num_nodes(ntype='user')

    return G, fea_dict, fea_dim_dict, usernum

def get_graph_eles(followfile,textvecfile,attrfile):
    adj = np.load(followfile)
    src = adj[:,0]
    dst = adj[:,1]
    with open(textvecfile,'rb') as f:
        usertextvec = pickle.load(f)
    usertextvec = sorted(usertextvec.items(),key=lambda item:item[0])
    usertextid = [uservec[0] for uservec in usertextvec]
    usertextfea = [uservec[1] for uservec in usertextvec]
    print("userfea file:", attrfile)
    userfea = np.load(attrfile)
    print(userfea)
    return src,dst,usertextid,usertextfea,userfea

def construct_joint_heter_graph_from_file(userfollowfileA,usertextvecfileA,userattrfileA,
                                          userfollowfileB,usertextvecfileB,userattrfileB,
                                          indexA,indexB):
    srcA,dstA,usertextidA,usertextfeaA,userfeaA = get_graph_eles(userfollowfileA,usertextvecfileA,userattrfileA)
    srcB, dstB, usertextidB, usertextfeaB,userfeaB = get_graph_eles(userfollowfileB, usertextvecfileB, userattrfileB)

    usernumA = np.max(srcA)+1 if np.max(srcA)>np.max(dstA) else np.max(dstA)+1
    srcB = srcB + np.array([usernumA]*srcB.shape[0])
    dstB = dstB + np.array([usernumA]*dstB.shape[0])
    src = np.concatenate([srcA,srcB],axis=0)
    dst = np.concatenate([dstA,dstB],axis=0)
    src = np.concatenate([indexA,src],axis=0)
    dst = np.concatenate([indexB + np.array([usernumA]*indexB.shape[0]), dst],axis=0)
    usertextidB = [id+usernumA for id in usertextidB]
    usertextid = usertextidA + usertextidB
    usertextfea = usertextfeaA + usertextfeaB
    userfea = np.concatenate([userfeaA,userfeaB],axis=0)
    print(userfea)

    G = dgl.heterograph({('user', 'hasfan', 'user'): (torch.tensor(dst), torch.tensor(src)),
                         ('user', 'follow', 'user'): (torch.tensor(src), torch.tensor(dst)),
                         # ('user','post','texts'):(torch.tensor(usertextid,dtype=torch.int32),
                         #                         torch.tensor(np.arange(len(usertextvec)),dtype=torch.int32)),
                         ('texts', 'self related', 'texts'): (
                         torch.tensor(np.arange(len(usertextfea)), dtype=torch.int32),
                         torch.tensor(np.arange(len(usertextfea)), dtype=torch.int32)),
                         ('texts', 'are posted by', 'user'): (
                             torch.tensor(np.arange(len(usertextfea)), dtype=torch.int32),
                             torch.tensor(usertextid, dtype=torch.int32))
                         })

    G.nodes['user'].data['fea'] = torch.tensor(userfea)
    G.nodes['texts'].data['fea'] = torch.tensor(usertextfea)
    fea_dict = dict()
    fea_dim_dict = dict()
    for ntype in G.ntypes:
        fea = G.nodes[ntype].data['fea']
        fea_dict[ntype] = fea
        fea_dim_dict[ntype] = fea.shape[1]
    usernumB = G.num_nodes(ntype='user') - usernumA
    print(usernumA,usernumB)
    return G, fea_dict, fea_dim_dict, usernumA,usernumB


def splitdata(labeldict, train_per, val_per):
    print(type(labeldict))
    if type(labeldict) == type(dict()):
        label_dict = labeldict
    else:
        with open(labeldict, 'rb') as f:
            label_dict = pickle.load(f)
    id = torch.tensor(list(label_dict.keys()),dtype=torch.int64)
    label = torch.tensor(list(label_dict.values()),dtype=torch.int64)
    print("num of nodes have label:", label.shape[0])

    posi = [i for i in range(0,id.shape[0])]
    random.seed(5)
    random.shuffle(posi)
    train_posi = torch.tensor(posi[0:int(id.shape[0]*train_per)])
    vali_posi = torch.tensor(posi[int(id.shape[0] * train_per)::])

    train_id = id[train_posi]
    train_label = label[train_posi]
    vali_id = id[vali_posi]
    vali_label = label[vali_posi]
    return train_id,train_label,vali_id,vali_label


def get_triplet(G, nodesnum, size):
    edges = torch.cat([G.edges(etype='hasfan')[0].unsqueeze(dim=1),
                       G.edges(etype='hasfan')[1].unsqueeze(dim=1)], dim=1)
    indexs = torch.randint(0, edges.shape[0], [size])
    pos = edges[indexs]
    negs = torch.randint(0, nodesnum, [size]).unsqueeze(dim=1)
    triplets = torch.cat([pos,negs], dim=1)
    return triplets

def get_triplet_from_edge_file(followingfile,nodesnum,size):
    edges = torch.IntTensor(np.load(followingfile))
    indexs = torch.randint(0,edges.shape[0],[size])
    pos = edges[indexs]
    negs = torch.randint(0,nodesnum,[size]).unsqueeze(dim=1)
    triplets = torch.cat([pos,negs],dim=1)
    return triplets

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx




def get_multi_adj(adj,nodes):
    N_adj = normalize(adj)
    N2_adj = np.matmul(N_adj, N_adj)
    N3_adj = np.matmul(N2_adj, N_adj)
    multi_adj = normalize(sparse.eye(nodes)+N_adj + N2_adj + N3_adj )
    return multi_adj