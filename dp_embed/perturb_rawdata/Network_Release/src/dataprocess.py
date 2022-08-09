import networkx as nx
from Network_Release.src.dataloader import Single_Graph_Dataset

def getnodedict(file):
    G = nx.Graph()
    index2user = dict()
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            usera, userb = line.strip().split()
            G.add_edge(usera,userb)
        adj = nx.to_scipy_sparse_matrix(G)
        for i,node in enumerate(G.nodes()):
            index2user[i] = node
        return adj,index2user


data_path = r"E:\federated\datasets\4\foursquare\users\user_following"
# data_path = r"E:\federated\datasets\4\twitter\following"
adj,index2user = getnodedict(data_path)
