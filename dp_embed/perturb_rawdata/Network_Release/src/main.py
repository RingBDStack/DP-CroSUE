import os
import pickle
import networkx as nx
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from Network_Release.src.logger import stat_logger
from Network_Release.src.DPGGAN.model import DPGGAN
# from src.DPGGAN.model import DPGGAN
from Network_Release.src.DPGGAN.adadp import ADADP
from Network_Release.src.config import load_config
from Network_Release.src.dataloader import Multi_Graph_Dataset,Single_Graph_Dataset
from Network_Release.src.train import train

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))

multi_graph_dataset = set(['foursquare','twitter'])

def arg_parser():
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43, help='Random seed.')
    parser.add_argument('--threads', type=int, default=4, help='Thread number.')
    parser.add_argument('--txt_log', type=bool, default=True, help='whether save the txt_log.')
    parser.add_argument('--model_name', type=str, default='DPGGAN', help='[DPGGAN, GGAN, DPGVAE, GVAE]')
    parser.add_argument('--dataset_str', type=str, default='foursquare')
    parser.add_argument('--n_eigenvector', type=int, default=128, help='use eigenvector as initial feature.')

    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--test_period', type=int, default=10, help='test period.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', default=0.01, help='the ratio of training set in whole dataset.')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--stat_generate_num', type=int, default=5, help='generate a batch of graph for graph stat.')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--discriminator_ratio', type=float, default=0.1, help='factor of discriminator loss.')
    parser.add_argument('--kl_ratio', type=float, default=1.0, help='factor of kl loss.')

    parser.add_argument('--std', type=float, default=0.75, help='Standard deviation of the Gaussian prior.')
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--draw_graph_epoch', type=int, default=2)
    parser.add_argument('--generated_graph_num', type=int, default=5, help='generate multiple graph at one time to compute variance.')
    parser.add_argument('--save_generated_graph', type=int, default=1, help='whether save generated graph or not.')

    args = parser.parse_args()
    return args



def set_env(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
    model_args = load_config(args.model_name, args.dataset_str, args.optimizer)
    if args.model_name in ['DPGVAE', 'GVAE']:
        args.discriminator_ratio = 0
    return args, model_args, current_time


def load_data(args, dataset_str):
    assert dataset_str in multi_graph_dataset
    args.dataset_type = 'multi'
    dataset = Multi_Graph_Dataset(dataset_str, args.n_eigenvector)
    args.num_samples = dataset.datasets[0].features.shape[0]
    return dataset

def load_single_data(dataset_str, n_eigenvector, adj):
    dataset = Single_Graph_Dataset(dataset_str, n_eigenvector, graph_adj=adj, label=None)
    return dataset

def save_data_path(data_dir, args):
    if 'DP' in args.model_name:
        save_path = data_dir + '{}_{}_eps:{}'.format(args.model_name, args.dataset_str, args.model_args.eps_requirement)
    else:
        save_path =data_dir + '{}_{}'.format(args.model_name, args.dataset_str)

    return save_path



def create_model(args, model_args, dataset):
    if  args.batch_size > dataset.features.shape[0]:
        args.batch_size = dataset.features.shape[0]
    model = None
    if args.model_name in ['GGAN', 'GVAE']:
        model = GGAN(args, model_args, dataset.features, dataset.adj_list)
    elif args.model_name in ['DPGGAN', 'DPGVAE']:
        model = DPGGAN(args, model_args, dataset.features, dataset.adj_list)
    else:
        print(args.model_name)
        print("Unknown model name.")
        exit(1)

        is_enc1 = True
        for v in model.parameters():
            if v is not None and v.requires_grad is True:
                if len(v.data.shape) == 2:
                    continue
                if is_enc1:
                    is_enc1 = False
                    v.data.copy_(v[0].data.clone().repeat(args.batch_size * (1 + model_args.samp_num), 1, 1))
                else:
                    v.data.copy_(v[0].data.clone().repeat(args.batch_size, 1, 1))
    return model


def create_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'ADADP':
        optimizer = ADADP(filter(lambda p: p.requires_grad, model.parameters()))
    return optimizer


def main(args, model_args, dataset, current_time):
    model = create_model(args, model_args, dataset)
    optimizer = create_optimizer(args, model)
    logger = stat_logger(args, current_time)
    print("begin train===")
    model, generated_adj, generated_graph_property_cache = train(args, model_args, dataset, model, optimizer, logger)
    return generated_adj, generated_graph_property_cache



def perturb_user_follow_graph(dataset_str,output_dir,n_eigenvector,adj):
    args = arg_parser() # args is general arguments
    args, model_args, current_time = set_env(args)
    args.model_args = model_args  # model_args is relevant to a specific model
    args.dataset_str = dataset_str
    args.n_eigenvector = n_eigenvector
    print("load_data")
    dataset = load_single_data(args.dataset_str,args.n_eigenvector, adj)
    args.num_samples = dataset.features.shape[0]

    model_args.dec2_dim = dataset.actual_feature_dim # the feat dim is capped for some small graph
    args.batch_size = dataset.adj.shape[0]
    generated_adj, generated_graph_property_cache = main(args, model_args, dataset, current_time)
    assert generated_adj.shape[0] == dataset.adj.shape[0]


    print("generated_adj:",generated_adj)
    s_path = save_data_path(output_dir,args)
    np.save(s_path, generated_adj)
    print("successfully save generated adj in %s"%s_path)
    # logger = stat_logger(args, current_time)
    # stat_log_info = logger.form_generated_stat_log('final', property_dict_all)
    # logger.write(stat_log_info)