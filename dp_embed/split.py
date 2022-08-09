import pandas as pd
from random import shuffle
import numpy as np
import pandas as pd
import pickle
def split_weibo(relfile):
    pass

if __name__ == "__main__":
    relfile = "../dataset/weibo/newrel.npy"
    edges = list(np.load(relfile))[0:20000]
    edges1 = edges[0:12000]
    edges2 = edges[8000:20000]
    user = list(np.array(edges)[:,0]) + list(np.array(edges)[:,1])
    user1 = list(np.array(edges1)[:,0]) + list(np.array(edges1)[:,1])
    user2 = list(np.array(edges2)[:, 0]) + list(np.array(edges2)[:, 1])
    user1 = set(user1)
    user2 = set(user2)

    userinfo = pd.DataFrame(pd.read_csv("../dataset/weibo/user_info.csv"))
    newuserinfo = userinfo.loc[userinfo['_id'].isin(user)]
    userinfo1 = userinfo.loc[userinfo['_id'].isin(user1)]
    userinfo2 = userinfo.loc[userinfo['_id'].isin(user2)]

    blog = pd.DataFrame(pd.read_csv("../dataset/weibo/blog.csv"))
    newblog = blog.loc[blog['user_id'].isin(user)]
    blog1 = blog.loc[blog['user_id'].isin(user1)]
    blog2 = blog.loc[blog['user_id'].isin(user2)]

    with open("../dataset/weibo/user_label.pkl", 'rb') as f:
        user_label = pickle.load(f)
    user_label1 = {}
    user_label2 = {}
    for u in user1:
        user_label1[u] = user_label[u]
    for u in user2:
        user_label2[u] = user_label[u]

    label1 = np.array(list(user_label1.values()))
    print(sum(label1==4))

    # with open("../dataset/weibo/user_label1.pkl", 'wb') as f:
    #     pickle.dump(user_label1,f)
    # with open("../dataset/weibo/user_label2.pkl", 'wb') as f:
    #     pickle.dump(user_label2,f)



    # np.save("../dataset/weibo/newrel1",edges1)
    # np.save("../dataset/weibo/newrel2", edges2)
    # newuserinfo.to_csv("../dataset/weibo/sub_user_info.csv")
    # userinfo1.to_csv("../dataset/weibo/user_info1.csv")
    # userinfo2.to_csv("../dataset/weibo/user_info2.csv")
    # blog1.to_csv("../dataset/weibo/blog1.csv")
    # blog2.to_csv("../dataset/weibo/blog2.csv")
    # newblog.to_csv("../dataset/weibo/subblog.csv")

    print(len(newuserinfo),len(userinfo1),len(userinfo2),len(blog1),len(blog2))



    print(len(user1),len(user2),len(user1&user2))



    # userfile = "../dataset/weibo/user_info.csv"
    # usercsv = pd.read_csv(userfile,error_bad_lines=False)
    # userdf = pd.DataFrame(usercsv)
    # userid = userdf['_id'].values.tolist()
    #
    # alluser = set()
    # for e in edges:
    #     alluser.add(e[0])
    #     alluser.add(e[1])
    # print("alluser:",len(alluser))
    #
    # edge1 = np.load("../dataset/weibo/newrel1.npy")
    # edge2 = np.load("../dataset/weibo/newrel2.npy")
    # all_edge = np.load("../dataset/weibo/newrel.npy")
    #
    # user1 = set(list(edge1[:, 0]) + list(edge1[:, 1]))
    # user2 = set(list(edge2[:, 0]) + list(edge2[:, 1]))
    # all_user = set(list(all_edge[:, 0]) + list(all_edge[:, 1]))
    #
    # blog_file = "../dataset/weibo/blog.csv"
    # blog_df = pd.DataFrame(pd.read_csv(blog_file))
    #
    # user_info_file = '../dataset/weibo/new_user_info.csv'
    # user_info_df = pd.DataFrame(pd.read_csv(user_info_file,error_bad_lines=False))
    # user_info_df.loc[user_info_df['_id'].isin(user2)].reset_index(drop=True).to_csv(
    #      '../dataset/weibo/user_info2.csv')
    #

    # blog_df.loc[blog_df['user_id'].isin(user1)].reset_index(drop=True).iloc[:,[1,2,3]].to_csv('../dataset/weibo/blog1.csv')
    # blog_df.loc[blog_df['user_id'].isin(user2)].reset_index(drop=True).iloc[:, [1, 2, 3]].to_csv(
    #     '../dataset/weibo/blog2.csv')

    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows', None)


