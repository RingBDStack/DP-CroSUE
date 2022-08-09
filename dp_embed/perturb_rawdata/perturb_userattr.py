import argparse
from simhash import Simhash
import numpy as np
from sklearn.preprocessing import normalize
from math import exp
import random
import pandas as pd
import math
import pickle

def getbit(byte,bits):
    result = []
    for index in range(bits):
        if(byte & (1<<(index))):
            result.append(1)
        else:
            result.append(0)
    return result

def get_four_attr(file,occufile,genderfile):
    #"../../dataset/foursquare/userocculabel.pkl"
    with open(occufile,'rb') as f:
        occu_dict = pickle.load(f)
        print("occu dict len:", len(occu_dict))
    with open(genderfile,'rb') as f:
        gender_dict = pickle.load(f)
        print("gender dict len:", len(gender_dict))
    nameattr = []
    otherattr = []
    occuattr = []
    genderattr  = []
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            text = line.split('\t')
            userhash = Simhash(text[0],f=32).value
            nameattr.append(getbit(userhash,32))
            if(len(text)<15):
                [text.append(0) for i in range(15-len(text))]
            for index in [12,13,14]:
                if text[index] == '':
                    text[index] = 0
            otherattr.append([text[12],text[13],text[14]])
            if text[0] in occu_dict:
                occuattr.append(occu_dict[text[0]])
            else:
                occuattr.append(np.zeros(18))
            if text[0] in gender_dict:
                if gender_dict[text[0]] == 0:
                    genderattr.append(np.array([1,0]))
                else:
                    genderattr.append(np.array([0,1]))
            else:
                genderattr.append(np.array([0,0]))
    otherattr = np.array(otherattr)
    otherattr = normalize(otherattr,axis=0,norm='max')
    nameattr = np.array(nameattr)
    occuattr = np.array(occuattr)
    userattr = np.concatenate((nameattr,otherattr),axis=1)
    userattr = np.concatenate((userattr,occuattr),axis=1)
    userattr = np.concatenate((userattr,genderattr),axis=1)
    return userattr

def get_weibo_attr(file,occufile,genderfile):
    with open(occufile,'rb') as f:
        occu_dict = pickle.load(f)
    with open(genderfile,'rb') as f:
        gender_dict = pickle.load(f)
    nameattr = []
    otherattr = []
    occuattr = []
    genderattr = []
    eles_list = pd.DataFrame(pd.read_csv(file)).loc[:,["NickName","Num_Tweets","Num_Follows","Num_Fans","_id"]].values.tolist()
    for eles in eles_list:
        userhash = Simhash(str(eles[0]),f=32).value
        nameattr.append(getbit(userhash,32))
        oattr = [0 if math.isnan(i) else i for i in eles[1:-1]]
        otherattr.append(oattr)
        if eles[-1] in occu_dict:
            occuattr.append(occu_dict[eles[-1]])
        else:
            occuattr.append(np.zeros(14))
        if eles[-1] in gender_dict:
            if gender_dict[eles[-1]] == 0:
                genderattr.append(np.array([1, 0]))
            else:
                genderattr.append(np.array([0, 1]))
        else:
            genderattr.append(np.array([0, 0]))
    otherattr = np.array(otherattr)
    otherattr = normalize(otherattr,axis=0,norm='max')
    nameattr = np.array(nameattr)
    userattr = np.concatenate((nameattr,otherattr),axis=1)
    occuattr = np.array(occuattr)
    userattr = np.concatenate((userattr,occuattr),axis=1)
    userattr = np.concatenate((userattr, genderattr), axis=1)
    return userattr

def get_twitter_attr(file,occufile,genderfile):
    #"../../dataset/twitter/userocculabel.pkl"
    with open(occufile,'rb') as f:
        occu_dict = pickle.load(f)
    with open(genderfile,'rb') as f:
        gender_dict = pickle.load(f)
    nameattr = []
    otherattr = []
    occuattr = []
    genderattr = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        combine = False
        former = ""
        for line in lines:
            if combine:
                line = former.strip() + ' '+line
                combine = False

            if len(line.strip().split('\t'))<7:
                combine = True
                former = line
                continue

            text = line.strip().split('\t')
            userhash = Simhash(text[0], f=32).value
            nameattr.append(getbit(userhash, 32))
            otherattr.append([text[-1].replace(',',''), text[-2].replace(',',''), text[-3].replace(',','')])

            if text[0] in occu_dict:
                occuattr.append(occu_dict[text[0]])
            else:
                occuattr.append(np.zeros(18))

            if text[0] in gender_dict:
                if gender_dict[text[0]] == 0:
                    genderattr.append(np.array([1, 0]))
                else:
                    genderattr.append(np.array([0, 1]))
            else:
                genderattr.append(np.array([0, 0]))
    otherattr = np.array(otherattr)
    otherattr = normalize(otherattr, axis=0, norm='max')
    nameattr = np.array(nameattr)
    occuattr = np.array(occuattr)
    userattr = np.concatenate((nameattr, otherattr), axis=1)
    userattr = np.concatenate((userattr,occuattr), axis=1)
    userattr = np.concatenate((userattr, genderattr), axis=1)
    return userattr

def perturb_single(x,eplison):
    p = exp(eplison/2.0)/(exp(eplison/2.0)+1.0)
    c = (exp(eplison/2.0)+1.0)/(exp(eplison/2.0)-1.0)
    lx = (c+1.0)/2.0*x - (c-1.0)/2
    pix = lx + c - 1.0
    if np.random.uniform()<p:
        x_ = np.random.uniform(lx,pix)
    else:
        if np.random.uniform()< (c-pix)/(c-pix+lx+c):
            x_ = np.random.uniform(-c,lx)
        else:
            x_ = np.random.uniform(pix,c)
    return x_

def perturb_data(attr, eplison, dimindex = 32, num = 2):
    total = attr.shape[1] - dimindex
    slice = random.sample([i for i in range(32)],num)
    slice += [dimindex+i for i in range(total-num)]
    for i in range(attr.shape[0]):
        for j in slice:
            attr[i,j] = perturb_single(attr[i,j],eplison*1.0/total)
    return attr


def make_user_attr(dataset, userfile, occufile, genderfile, eplison, ori_save_file, per_save_file):
    if dataset == "foursquare":
        ori_attr = get_four_attr(userfile,occufile,genderfile)
    elif dataset == "twitter":
        ori_attr = get_twitter_attr(userfile,occufile,genderfile)
    elif dataset == "weibo1" or dataset == "weibo2":
        ori_attr = get_weibo_attr(userfile,occufile,genderfile)
    np.save(ori_save_file, ori_attr)
    print("attr shape:",ori_attr.shape)
    print("successfully save %s ori_attr in %s"%(dataset,ori_save_file))
    per_attr = perturb_data(ori_attr, eplison)
    np.save(per_save_file.format(str(eplison)),per_attr)
    print("successfully save %s perturbed_attr in %s"%(dataset,per_save_file))


