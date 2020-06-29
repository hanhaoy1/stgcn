from data_loader import read_dataset, load_data
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import geohash

def poi2poi(dataset):
    data = read_dataset(dataset)
    data = data.groupby(['uid']).filter(lambda x: len(x) >= 10)
    grouped = data.groupby(['uid']).apply(lambda x: x.sort_values('time'))
    poi_pairs = defaultdict(int)
    timedelta = datetime.timedelta(days=7)
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
    with open('./dataset/'+dataset+'/poi2poi1.txt', 'w') as f:
        for k, v in poi_pairs.items():
            f.write(str(k[0]) + '\t' + str(k[1]) + '\t' + str(v) + '\n')


def region_neighbors(dataset):
    data = load_data(dataset)
    region_pairs = set()
    regions = set(data['region'])
    if dataset == 'meituan':
        regions = regions.union(set(data['user_region']))
    for region in regions:
        neighbors = geohash.neighbors(region)
        for neighbor in neighbors:
            if neighbor in regions:
                if neighbor < region:
                    region_pairs.add((neighbor, region))
                else:
                    region_pairs.add((region, neighbor))
    with open('./dataset/'+dataset+'/region.txt', 'w') as f:
        for r1, r2 in region_pairs:
            f.write(r1 + '\t' + r2 + '\n')


if __name__ == '__main__':
    dataset = 'foursquare'
    poi2poi(dataset)
