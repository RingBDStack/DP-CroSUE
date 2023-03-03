from transformers import BertTokenizer, BertForMaskedLM
import en_core_web_lg
import numpy as np
import pickle
import argparse
import pandas as pd
import zh_core_web_lg

class textdata():
    def __init__(self,dataset,textfilename,utfilename):
        self.dataset = dataset
        self.textfile = textfilename
        self.usertextfile = utfilename
        self.readidcont()
        self.readusercont()
        self.transcont2vec()

    def readidcont(self):
        if self.dataset=="foursquare" or self.dataset=="twitter":
            print("----------",'read idcont----------')
            with open(self.textfile, 'r', encoding='ISO-8859-1') as f:
                texts = f.readlines()
            self.idcont = dict()
            count = 0
            for i,text in enumerate(texts):
                if (len(text.strip().split('\t\t')) == 1):
                    continue
                id = text.strip().split('\t\t')[0]
                cont = text.strip().split('\t\t')[1]
                self.idcont[id] = cont
                count +=1
            print("total %s texts num:"%self.dataset, count)

    def readusercont(self):
        print("----------", 'read usercont----------')
        if self.dataset == "twitter" or self.dataset=="foursquare":
            with open(self.usertextfile, 'r') as f:
                texts = f.readlines()
            self.usercont = dict()
            for text in texts:
                u, id = text.strip().split()
                if u in self.usercont.keys():
                    if id in self.idcont:
                        self.usercont[u] += self.idcont[id]
                else:
                    if id in self.idcont:
                        self.usercont[u] = self.idcont[id] + " "
        elif self.dataset == "weibo1" or self.dataset == "weibo2":
            userlist = pd.DataFrame(pd.read_csv(self.usertextfile))["_id"].values.tolist()
            blogdf = pd.DataFrame(pd.read_csv(self.textfile))
            self.usercont = dict()
            for user in userlist:
                blog = ""
                for b in blogdf.loc[blogdf['user_id'] == user]["con"]:
                    if len(blog) < 1000:
                        blog = blog + " " + b
                    else:
                        break
                self.usercont[user] = blog.lstrip()



    def transcont2vec(self):
        print("----------", 'transcont2vec----------')
        tokenizer = en_core_web_lg.load()
        if self.dataset=="weibo1" or self.dataset == "weibo2":
            tokenizer = zh_core_web_lg.load()
        self.usercontvec = dict()
        length = len(self.usercont)
        for i,user in enumerate(self.usercont.keys()):
            if(len(self.usercont[user].split())>2000):
                self.usercont[user] = " ".join(list(self.usercont[user].split())[0:2000])
            self.usercontvec[user] = tokenizer(self.usercont[user]).vector
            if i%10000==0:
                print(i,"%",length)


    def getusercont(self):
        return self.usercont

    def getusercontvec(self):
        return self.usercontvec










def make_user_text_vec(dataset,textfile,utfile,outfile):
    usercontvec = textdata(dataset, textfile, utfile).getusercontvec()
    with open(outfile + '.pkl', 'wb') as of:
        pickle.dump(usercontvec, of)
        print("successfully save usertextvec to file %s"%outfile+'.pkl')

