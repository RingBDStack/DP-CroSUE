import dgl
import dgl.function as fn
import torch
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops

ms.set_context(device_target='GPU', device_id=0)
g_data = {
    ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([1,
                                                                        2])),
    ('drug', 'treat', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
}

g = dgl.heterograph(g_data)
print(g)
print(g.edges['interacts'])
g.nodes['drug'].data['hid'] = torch.tensor([[1, 1], [1, 1]],
                                           dtype=torch.float32)
g.nodes['gene'].data['hid'] = torch.tensor([[2, 3], [2, 3], [2, 3], [2, 3]],
                                           dtype=torch.float32)
funcs = {}
print(g.etypes)
for etype in g.etypes:
    funcs[etype] = (fn.copy_u('hid', 'm'), fn.mean('m', 'h'))
g.multi_update_all(funcs, 'sum')
print(g.nodes['drug'])
print(g.nodes['gene'])

nodeType_list = g.ntypes
node_dict = {
    'drug': Tensor(g.nodes('drug').numpy(), ms.int32),
    'gene': Tensor(g.nodes('gene').numpy(), ms.int32)
}
print(node_dict['drug'])
edgeType_list = g.etypes
edge_dict = {}
for e in edgeType_list:
    edge_dict[e] = {
        'srctype':
        g.to_canonical_etype(e)[0],
        'dsttype':
        g.to_canonical_etype(e)[2],
        'edges':
        Tensor([g.edges(etype=e)[0].numpy(),
                g.edges(etype=e)[1].numpy()], ms.int32)
    }

hidden = {
    'drug': Tensor([[1, 1], [1, 1]], ms.float32),
    'gene': Tensor([[[2, 3], [2, 3], [2, 3]]], ms.float32)
}
h = {}
for ntype in nodeType_list:
    h[ntype] = ops.Zeros()((node_dict[ntype].shape[0], 2), ms.int32)
for etype in edgeType_list:
    srctype = edge_dict[etype]['srctype']
    dsttype = edge_dict[etype]['dsttype']
    edges = edge_dict[etype]['edges']
    e_m = hidden[srctype][edges[0]]
    mailbox = ops.Zeros()((node_dict[dsttype].shape[0], 2), ms.float32)
    print(mailbox.__class__)
    mailbox = ops.tensor_scatter_add(mailbox, edges[1].reshape(-1, 1), e_m)
    print(mailbox)
    # test = ops.scatter_add(mailbox, edges[1], e_m)
    # cnt = ops.tensor_scatter_add(
    #     ops.Zeros()((node_dict[dsttype].shape[0], 2), ms.int32),
    #     edges[1].astype(ms.int32).reshape(-1, 1),
    #     ops.Ones()(e_m.shape, ms.int32))
    cnt = ops.scatter_add(
        ops.Zeros()((node_dict[dsttype].shape[0], 2), ms.int32),
        edges[1].astype(ms.int32),
        ops.Ones()(e_m.shape, ms.int32))
    mailbox = mailbox / ops.maximum(cnt, Tensor(1, ms.int32))
    h[dsttype] += mailbox
    # h[ntype][i] += mailbox[ntype][i]
# for ntype in nodeType_list:
#     for i in range(node_dict[ntype].shape[0]):
#         if len(mailbox[ntype][i]) > 0:
#             h[ntype][i] += mailbox[ntype][i]
print(h)
