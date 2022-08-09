import argparse
import os
from perturb_userattr import make_user_attr
from San_Text.run_SanText import make_santext
from generate_text_feature import make_user_text_vec
from convert2id import convert_userfollowing
from convert2id import getuser2id, convert_userfollowing, convert_label_textvec
import dgl
import numpy as np
import torch
# from Network_Release.src.main import perturb_user_follow_graph
from scipy import sparse
from perturb_adj import perturb_edges


def arg_parser():
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo1', help='[foursquare, twitter]')
    parser.add_argument('--exp_folder', type=str, default="")
    parser.add_argument('--ori_data_folder', type=str, default="")
    parser.add_argument('--perturb_data_folder', type=str, default="")
    # user attribute
    parser.add_argument('--user_attr_file', type=str, default="")
    parser.add_argument('--attr_epsilon', type=str, default="")
    # user texts
    parser.add_argument('--text_data_dir', type=str, default="")
    parser.add_argument('--text_data_file', type=str, default="")
    parser.add_argument('--text_epsilon', type=int, default=1)
    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument("--sensitive_word_percentage", type=float, default=1)
    # generate text vector
    parser.add_argument('--usertextfile', type=str, default="")
    parser.add_argument('--text_file', type=str, default="")
    parser.add_argument('--san_text_file', type=str, default="")
    parser.add_argument('--out_textvec_file', type=str, default="")
    parser.add_argument('--out_santextvec_file', type=str, default="")
    # generate follow graph
    parser.add_argument('--user_follow_file', type=str, default="")
    parser.add_argument('--n_eigenvector', type=int, default=128, help='use eigenvector as initial feature.')
    args = parser.parse_args()
    return args


def set_data_args(dataset, args):
    if dataset == "foursquare":
        args.dataset = "foursquare"
        args.exp_folder = "../../dataset/foursquare/"
        args.ori_data_folder = args.exp_folder + "data/origin/"
        args.perturb_data_folder = args.exp_folder + "data/perturbed/"
        # user attribute
        args.user_attr_file = args.exp_folder + "users/user"
        args.save_ori_attr_path = args.ori_data_folder + "userattr"
        args.attr_epsilon = 5
        args.save_perturb_attr_path = args.perturb_data_folder + "userattr-{}".format(str(args.attr_epsilon))
        # user texts
        args.text_data_dir = "../../dataset/foursquare/tips"
        args.text_data_file = "../../dataset/foursquare/tips/tipcleaned"
        args.text_epsilon = 5
        args.p = 0.2
        args.sensitive_word_percentage = 1
        # user texts vector
        args.usertextfile = "../../dataset/foursquare/tips/user_tip"
        args.text_file = "../../dataset/foursquare/tips/tipcleaned"
        args.san_text_file = "../../dataset/foursquare/tips/" + "text_eps_%.2f" % args.text_epsilon + "sword_%.2f_p_%.2f" % (
            args.sensitive_word_percentage, args.p)
        args.out_textvec_file = args.ori_data_folder + "user_text_vec"
        args.out_santextvec_file = args.perturb_data_folder + "san_user_text_vec_%.2f_%.2f_%.2f" % (
        args.text_epsilon, args.sensitive_word_percentage, args.p)
        # userfollow
        args.follow_eplison = 15
        args.user_follow_file = "../../dataset/foursquare/users/user_following"
        args.label_path = args.exp_folder + "user_label.pkl"
        args.multi_label_path = args.exp_folder + "jointmultilabel1.pkl"
        args.label_dir = args.exp_folder + "data/"
        args.gender_label_path = args.exp_folder + "genderlabel.pkl"
        args.occu_label_path = args.exp_folder + "userocculabel.pkl"
    elif dataset == "twitter":
        args.dataset = "twitter"
        args.exp_folder = "../../dataset/twitter/"
        args.ori_data_folder = args.exp_folder + "data/origin/"
        args.perturb_data_folder = args.exp_folder + "data/perturbed/"
        # user attribute
        args.user_attr_file = args.exp_folder + "user"
        args.save_ori_attr_path = args.ori_data_folder + "userattr"
        args.attr_epsilon = 5
        args.save_perturb_attr_path = args.perturb_data_folder + "userattr-{}".format(str(args.attr_epsilon))

        # user texts
        args.text_data_dir = "../../dataset/twitter"
        args.text_data_file = "../../dataset/twitter/tweetcleaned"
        args.text_epsilon = 5
        args.p = 0.2
        args.sensitive_word_percentage = 1
        # user texts vector
        args.usertextfile = "../../dataset/twitter/userTweet"
        args.text_file = "../../dataset/twitter/tweetcleaned"
        args.san_text_file = "../../dataset/twitter/" + "text_eps_%.2f" % args.text_epsilon + "sword_%.2f_p_%.2f" % (
            args.sensitive_word_percentage, args.p)
        args.out_textvec_file = args.ori_data_folder + "user_text_vec"
        args.out_santextvec_file = args.perturb_data_folder + "san_user_text_vec_%.2f_%.2f_%.2f" % (
            args.text_epsilon, args.sensitive_word_percentage, args.p)
        # userfollow
        args.user_follow_file = "../../dataset/twitter/following"
        args.label_path = args.exp_folder + "user_label.pkl"
        args.follow_eplison = 15
        args.multi_label_path = args.exp_folder + "/jointmultilabel1.pkl"
        args.label_dir = args.exp_folder + "data/"
        args.gender_label_path = args.exp_folder + "genderlabel.pkl"
        args.occu_label_path = args.exp_folder + "userocculabel.pkl"
    elif dataset == "weibo1" or dataset == "weibo2":
        args.dataset = dataset
        args.exp_folder = "../../dataset/weibo/"
        args.ori_data_folder = args.exp_folder + "data/%s/origin/" % dataset
        args.perturb_data_folder = args.exp_folder + "data/%s/perturbed/" % dataset
        # user attribute
        args.user_attr_file = args.exp_folder + "user_info" + dataset[-1] + ".csv"
        args.save_ori_attr_path = args.ori_data_folder + "userattr"
        args.attr_epsilon = 5
        args.save_perturb_attr_path = args.perturb_data_folder + "userattr-{}".format(str(args.attr_epsilon))
        # user texts
        args.text_data_dir = "../../dataset/weibo"
        args.text_data_file = "../../dataset/weibo/cleaned_blog%s.csv" % dataset[-1]
        args.text_epsilon = 0.1
        args.p = 0.2
        args.sensitive_word_percentage = 1
        # user texts vector
        args.text_file = "../../dataset/weibo/cleaned_blog%s.csv" % dataset[-1]
        args.usertextfile = "../../dataset/weibo/user_info%s.csv" % dataset[-1]
        args.san_text_file = "../../dataset/weibo/" + "blog%s_eps_%.2f" % (
        dataset[-1], args.text_epsilon) + "sword_%.2f_p_%.2f.csv" % (
                                 args.sensitive_word_percentage, args.p)
        args.out_textvec_file = args.ori_data_folder + "user_text_vec"
        args.out_santextvec_file = args.perturb_data_folder + "san_user_text_vec_%.2f_%.2f_%.2f" % (
            args.text_epsilon, args.sensitive_word_percentage, args.p)
        # userfollow
        args.follow_eplison = 15
        args.user_follow_file = "../../dataset/weibo/newrel" + dataset[-1] + ".npy"
        args.label_path = args.exp_folder + "/user_label%s.pkl" % dataset[-1]
        args.multi_label_path = args.exp_folder + "/multilabel.pkl"
        args.label_dir = args.exp_folder + "data/" + dataset + "/"
        args.occu_label_path = args.exp_folder + "user_occu_label{}.pkl".format(dataset[-1])
        args.gender_label_path = args.exp_folder + "user_gender_label{}.pkl".format(dataset[-1])

    else:
        raise RuntimeError(f"dataset not supported")

    if not os.path.exists(args.exp_folder + "data"):
        os.makedirs(args.exp_folder + "data")
    if not os.path.exists(args.ori_data_folder):
        os.makedirs(args.ori_data_folder)
        os.makedirs(args.perturb_data_folder)

    return args


def make_data_embedding(dataset):
    args = arg_parser()
    print(args.dataset)
    args = set_data_args(dataset, args)
    user2id = getuser2id(args.dataset, args.user_attr_file)

    # make original user attribute features and perturbed features
    if not os.path.exists(args.save_perturb_attr_path + '.npy'):
        print("generate %s user attribute--------" % dataset)
        make_user_attr(args.dataset, args.user_attr_file, args.occu_label_path, args.gender_label_path,
                       args.attr_epsilon,
                       args.save_ori_attr_path, args.save_perturb_attr_path)
    else:
        print("orginal user attribute file:", args.save_ori_attr_path + '.npy')
        print("perturbed user attribute file:", args.save_perturb_attr_path + '.npy', '\n')

    # perturb texts
    if not os.path.exists(args.san_text_file):
        print("perturb %s texts------" % dataset)
        make_santext(args.dataset, args.text_data_dir, args.text_data_file, args.text_epsilon, args.p, args.sensitive_word_percentage,args.san_text_file)
    else:
        print('san_text_file:',args.san_text_file,'\n')

    # make original user text features and perturbed features
    if not os.path.exists(args.out_textvec_file + '.pkl'):
        print("generate %s original user text vector-------" % (args.dataset))
        make_user_text_vec(args.dataset, args.text_file, args.usertextfile, args.out_textvec_file)
        convert_label_textvec(user2id, args.out_textvec_file + '.pkl', args.ori_data_folder)

    else:
        print('original user text vector file:', args.out_textvec_file + '.pkl')
        convert_label_textvec(user2id, args.out_textvec_file + '.pkl', args.ori_data_folder)

    if not os.path.exists(args.out_santextvec_file + '.pkl'):
        print("generate %s perturbed user text vector-------" % (args.dataset))
        make_user_text_vec(args.dataset, args.san_text_file, args.usertextfile, args.out_santextvec_file)
        convert_label_textvec(user2id, args.out_santextvec_file + '.pkl', args.perturb_data_folder)

    # else:
    #     print("perturbed user text vector file:",args.out_santextvec_file+'.pkl','\n')
    #     convert_label_textvec(user2id, args.out_santextvec_file + '.pkl', args.perturb_data_folder)

    if not os.path.exists(args.ori_data_folder + 'following.npy'):
        convert_userfollowing(args.dataset, user2id, args.user_follow_file, args.ori_data_folder)
        edges = np.load(args.ori_data_folder + 'following.npy')
    else:
        edges = np.load(args.ori_data_folder + 'following.npy')
        print("original user following graph edge num:", edges.shape[0])

    # perturb user following graph
    # print("perturb user following graph-------")
    # perturb_edges(edges, args.perturb_data_folder, args.follow_eplison)
    # new_edges = np.load(args.perturb_data_folder + "tmf_eps:%s.npy" % str(args.follow_eplison))
    # print("new user following graph edge num:", new_edges.shape[0])

    # print(args.multi_label_path)
    # assert os.path.exists(args.multi_label_path)
    # convert_label_textvec(user2id,args.label_path,args.label_dir)
    # convert_label_textvec(user2id, args.gender_label_path, args.label_dir)
    # convert_label_textvec(user2id,args.occu_label_path, args.label_dir)


make_data_embedding("foursquare")
