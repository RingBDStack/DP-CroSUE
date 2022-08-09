import numpy as np
import math
import random


def top_m_filter(edge_list,node_num,edge_num,eplison1,eplison2=1):
    n = node_num
    m = edge_num
    m_ = m + int(np.random.laplace(0,1.0/eplison2))
    print(m_)
    eplison_t = math.log(n*(n-1)/(2*m_)-1)
    if eplison1<eplison_t:
        theta = 1/(2*eplison1)*eplison_t
    else:
        theta = 1/eplison1*math.log(n*(n-1)/(4*m_+1/2*(math.exp(eplison1)-1)))

    n1 = 0
    new_edge_list = []
    for edge in edge_list:
        w = 1 + np.random.laplace(0,1.0/eplison1)
        if w > theta:
            new_edge_list.append(edge)
            n1 += 1
    print(n1)
    while n1 < m_-1:
        src = random.randint(0,node_num-1)
        dst = random.randint(0,node_num-1)
        edge = (src,dst)
        if edge not in new_edge_list:
            new_edge_list.append(edge)
            n1+=1
    new_edge_list.append((node_num-1,node_num-1))
    print("new_edge:",new_edge_list)
    return np.array(new_edge_list)



def perturb_edges(edges, folder, eplison):
    node_num = np.max(edges) + 1
    edge_num = edges.shape[0]
    src = list(edges[:,0])
    tgt = list(edges[:,1])
    edges = list(zip(src,tgt))
    new_edges = top_m_filter(edges, node_num, edge_num, eplison)
    np.save(folder + 'tmf_eps:'+ str(eplison), new_edges)
    print("save following path:",folder + 'tmf_eps:'+ str(eplison))


