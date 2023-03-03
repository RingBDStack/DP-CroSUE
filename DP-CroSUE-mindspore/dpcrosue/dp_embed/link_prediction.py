import numpy as np
import mindspore
from utils import construct_heter_graph_from_file
from mindspore import Tensor
from model.HGNN import HeteroGCN

mindspore.set_context(device_target='GPU', device_id=0)


def get_edge_label(file):
    edges = np.load(file)
    id_label = dict()
    nodes_num = np.max(edges)
    for src, dst in list(edges):
        if src in id_label:
            id_label[src].append(dst)
        else:
            id_label[src] = [dst]

        if dst in id_label:
            id_label[dst].append(src)
        else:
            id_label[dst] = [src]
    labels = sorted(id_label.items(), key=lambda k: k[0])
    labels = [l[1] for l in labels]
    return labels, nodes_num


def predict_from_embedding(file):
    if file.split('.')[-1] == "npy":
        emb = np.load(file)
    elif file.split('.')[-1] == "ckpt":
        emb_model = HeteroGCN(fea_dim_dict, 256, 128)
        mindspore.load_checkpoint(file, net=emb_model)
        nodeType_list = G.ntypes
        edgeType_list = G.etypes
        node_dict = {
            e: Tensor(G.nodes(e).numpy(), mindspore.int32)
            for e in nodeType_list
        }
        edge_dict = {}
        for e in edgeType_list:
            edge_dict[e] = {
                'edges':
                Tensor(
                    [G.edges(etype=e)[0].numpy(),
                     G.edges(etype=e)[1].numpy()], mindspore.int32)
            }
        for e in fea_dict.keys():
            fea_dict[e] = Tensor(fea_dict[e].numpy(), mindspore.float32)
        emb = emb_model(node_dict, edge_dict, fea_dict)
        emb = emb.asnumpy()

    score = np.dot(emb, emb.T)
    value = np.linalg.norm(emb, axis=1, keepdims=True)
    value_matrix = np.dot(value, value.T)
    norm_score = score / value_matrix
    data = np.argsort(-norm_score, axis=1)
    pred = data[:, 1:4]
    return pred


if __name__ == "__main__":
    following_file = "../dataset/foursquare/data/origin/following.npy"
    textvecfile = "../dataset/foursquare/data/origin/user_text_vec2id.pkl"
    attfile = "../dataset/foursquare/data/origin/userattr.npy"
    emb_model_file = "../dataset/foursquare/data/origin/useremb.ckpt"
    G, fea_dict, fea_dim_dict, usernum = construct_heter_graph_from_file(
        following_file, textvecfile, attfile)
    pred = predict_from_embedding(emb_model_file)
    labels, num = get_edge_label(following_file)
    count = 0
    for l, p in zip(pred, labels):
        if len(set(l) & set(p)) > 0:
            count += 1
    print(1.0 * count / num)
