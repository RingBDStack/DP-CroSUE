import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.special import softmax
import pandas as pd
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def readfourloc(userfile, locfile):
    users = []
    tipids = []
    locs = []
    indexs = []
    with open(userfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            usertipid = line.strip().split()[0]
            eles = list(usertipid.split('/'))
            tipids.append(eles[-1])
            del eles[-1]
            users.append("".join(eles))

    with open(locfile, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if len(line.strip().split()) == 2:
                loc = list(map(float, list(line.strip().split())))
                locs.append(loc)
                indexs.append(int(index))
    users = np.array(users)
    tipids = np.array(tipids)
    locs = np.array(locs)
    indexs = np.array(indexs)
    return users[indexs], tipids[indexs], locs


def cal_probability(loc1, loc2, epsilon=2.0):
    distance = euclidean_distances(loc1, loc2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def santextloc(prob_matrix, locs):
    newindex = []
    for index in range(locs.shape[0]):
        sampling_prob = prob_matrix[index]
        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        newindex.append(sampling_index[0])
    newlocs = locs[np.array(newindex)]
    return newlocs


def changefourloc():
    userfile = "../../dataset/foursquare/locations/location"
    locfile = "../../dataset/foursquare/locations/loc_lat_lon"
    outfile = "../../dataset/foursquare/locations/newlocation1"

    if not os.path.exists(outfile):
        users, tipids, locs = readfourloc(userfile, locfile)
        print(np.where((locs[:, 0] < 40))[0].shape)
        position = np.array(np.where((locs[:, 0] < 40))[0])

        logger.info("Calculating Prob Matrix for Exponential Mechanism...")
        prob_matrix = cal_probability(locs[position], locs[position])
        logger.info("get new loc...")
        newlocs = santextloc(prob_matrix, locs[position])

        locdata = {"user": list(users[position]), "tipid": list(tipids[position]), "loc": list(newlocs)}
        locdata = pd.DataFrame(data=locdata, columns=['user', 'tipid', 'loc'])
        logger.info(locdata)
        np.save(outfile, locdata)
    else:
        locdata = np.load(outfile + ".npy", locfile)
        locdata = pd.DataFrame(data=locdata, columns=['user', 'tipid', 'loc'])
        logger.info(locdata)

