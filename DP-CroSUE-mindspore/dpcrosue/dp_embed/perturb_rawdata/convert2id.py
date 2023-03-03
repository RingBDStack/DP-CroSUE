import numpy as np
import pickle
import pandas as pd
import os
import dgl
def getfour(userfile):
    with open(userfile,'r',encoding='utf-8') as f:
        user2id = dict()
        lines = f.readlines()
        for i, line in enumerate(lines):
            user = line.strip().split('\t')[0]
            user2id[user] = i
    return user2id

def getweibo(userfile):
    users = pd.DataFrame(pd.read_csv(userfile)).loc[:,"_id"].values.tolist()
    #print(users)
    user2id = dict(zip(users,list(range(0,len(users)))))
    print("num",len(user2id))
    return user2id

def gettwitter(userfile):
    with open(userfile, 'r', encoding='utf-8') as f:
        user2id = dict()
        lines = f.readlines()
        combine = False
        all = ""
        count = 0
        for line in lines:
            if (combine):
                line = all + line
                all = ""
                combine = False

            line_list = line.strip().split('\t')
            if (len(line_list) < 7):
                combine = True
                all = line
                continue
            user2id[line_list[0]] = count
            count+=1
    return user2id




def getuser2id(dataset,userfile):
    if dataset == "foursquare":
        user2id = getfour(userfile)
    elif dataset == "twitter":
        user2id = gettwitter(userfile)
    elif dataset == "weibo1" or dataset == "weibo2":
        user2id = getweibo(userfile)
    else:
        raise RuntimeError(f"dataset not supported")
    print("dataset:",dataset,"usernum:",len(user2id))
    return user2id


def convert_userfollowing(dataset,user2id,followingfile,outdir):
    newfile = outdir + "following"
    userlist = []
    if dataset == "foursquare" or dataset=="twitter":
        with open(followingfile,'r') as f:
            lines = f.readlines()
            for line in lines:
                usera, userb = line.strip().split()
                userlist.append([user2id[usera],user2id[userb]])
    else:
        lines = np.load(followingfile)
        for line in lines:
            userlist.append([user2id[line[0]],user2id[line[1]]])
    edges = np.array(userlist)
    np.save(newfile,edges)
    print("successfully save original following graph in {}.npy".format(newfile))

def convert_label_textvec(user2id,srcfile,outdir):
    newfile = (outdir + srcfile.split('/')[-1]).replace('.pkl','') + "2id.pkl"
    with open(srcfile,'rb') as f:
        srcdict = pickle.load(f)
    print("len src dict:",len(srcdict))
    print("len user id:",len(user2id))
    id2value = dict()
    for user in srcdict.keys():
        if user in user2id:
            id2value[user2id[user]] = srcdict[user]
    with open(newfile,'wb') as f:
        pickle.dump(id2value,f)
    print("len label:", len(id2value))
    print("successfully save {}".format(newfile))
