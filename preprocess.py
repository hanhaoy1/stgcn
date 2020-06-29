from data_loader import read_dataset, load_data
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import geohash
import json

def poi2poi(data, days=7):
    # data = data.groupby(['uid']).filter(lambda x: len(x) >= 10)
    grouped = data.groupby(['uid']).apply(lambda x: x.sort_values('time'))
    poi_pairs = defaultdict(int)
    # timedelta = datetime.timedelta(hours=2)
    timedelta = datetime.timedelta(days=days)
    for idx, v in grouped.groupby(level=[0]):
        times = v['time'].tolist()
        for i in range(len(times)-1):
            for j in range(i+1, len(times)):
                if times[j] - times[i] < timedelta:
                    poi1 = v.iloc[i]['pid']
                    poi2 = v.iloc[j]['pid']
                    if poi1 < poi2:
                        poi_pairs[(poi1, poi2)] += 1
                    elif poi1 > poi2:
                        poi_pairs[(poi2, poi1)] += 1
                else:
                    break
    return poi_pairs


def region_neighbors(data, user_region=True):
    region_pairs = set()
    regions = set(data['region'])
    if user_region:
        regions = regions.union(set(data['user_region']))
    for region in regions:
        neighbors = geohash.neighbors(region)
        for neighbor in neighbors:
            if neighbor in regions:
                if neighbor < region:
                    region_pairs.add((neighbor, region))
                else:
                    region_pairs.add((region, neighbor))
    return regions, region_pairs


def process(dataset, train_ratio=0.8, user_region=True):
    data = load_data(dataset)
    regions, region_pairs = region_neighbors(data, user_region=user_region)
    users = set(data['uid'])
    pois = set(data['pid'])
    num_users = len(users)
    num_pois = len(pois)
    user2id = {v: k for k, v in enumerate(users)}
    poi2id = {v: k + num_users for k, v in enumerate(pois)}
    region2id = {v: k + num_users + num_pois for k, v in enumerate(regions)}
    with open('./dataset/' + dataset + '/user2id.json', 'w') as f:
        f.write(json.dumps(user2id))
    with open('./dataset/' + dataset + '/poi2id.json', 'w') as f:
        f.write(json.dumps(poi2id))
    with open('./dataset/' + dataset + '/region2id.json', 'w') as f:
        f.write(json.dumps(region2id))
    data['uid'] = data['uid'].apply(lambda x: user2id[x])
    data['pid'] = data['pid'].apply(lambda x: poi2id[x])
    data['region'] = data['region'].apply(lambda x: region2id[x])
    region_pairs = [[region2id[i], region2id[j]] for i, j in region_pairs]
    if user_region:
        data['user_region'] = data['user_region'].apply(lambda x: region2id[x])
    msk = np.random.rand(len(data)) < train_ratio
    train = data[msk]
    test = data[~msk]
    train.to_csv('./dataset/' + dataset + '/train.csv')
    test.to_csv('./dataset/' + dataset + '/test.csv')
    poi_pairs_train = poi2poi(train)
    user_poi_train = train.groupby(['uid', 'pid']).size().reset_index(name='w')
    poi_region = data[['pid', 'region']].drop_duplicates()
    poi_region['w'] = 1
    with open('./dataset/' + dataset + '/region2region.txt', 'w') as f:
        for r1, r2 in region_pairs:
            f.write(str(r1) + '\t' + str(r2) + '\t1\n')
    with open('./dataset/' + dataset + '/poi2poi_train.txt', 'w') as f:
        for k, v in poi_pairs_train.items():
            p1 = str(k[0])
            p2 = str(k[1])
            w = str(v)
            f.write(p1 + '\t' + p2 + '\t' + w + '\n')
    path = './dataset/' + dataset + '/user_poi_train.txt'
    user_poi_train.to_csv(path, sep='\t', header=False, index=False)
    path = './dataset/' + dataset + '/poi_region.txt'
    poi_region.to_csv(path, sep='\t', header=False, index=False)


if __name__ == '__main__':
    # process('meituan')
    process('gowalla', user_region=False)




