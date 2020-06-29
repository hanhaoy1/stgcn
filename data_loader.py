import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import geohash
import os

def gowalla_to_gps(location):
    l = location.split(',')
    la, lo = map(float, l[:2])
    return [la, lo]

def foursquare_to_gps(location):
    l = location[1:-1].split(',')
    try:
        la, lo = map(float, l[:2])
    except:
        return None
    return [la, lo]


def meituan_to_gps(location):
    l = location.strip().split(':')
    try:
        la, lo = float(l[1]), float(l[0])
    except:
        return None
    return [la, lo]


def read_dataset(dataset):
    if dataset == 'meituan':
        pass
    elif dataset == 'foursquare':
        date_col = ['Time(GMT)']
        data = pd.read_csv('./dataset/foursquare/foursquare.csv', sep='\t',
                           usecols=['userID', 'Time(GMT)', 'VenueId', 'VenueLocation'], parse_dates=date_col,
                           infer_datetime_format=True)
        data['gps'] = data['VenueLocation'].apply(foursquare_to_gps)
        data = data.rename(columns={'userID': 'uid', 'Time(GMT)': 'time', 'VenueId': 'pid'})
        data = data.drop(columns=['VenueLocation'])
        data = data[data['gps'].notnull()]
        data = data.reset_index(drop=True)
        data = data[['uid', 'pid', 'time', 'gps']]
    elif dataset == 'gowalla':
        date_col = ['datetime']
        data = pd.read_csv('./dataset/gowalla/gowalla_food.csv', sep='\t', parse_dates=date_col, infer_datetime_format=True)
        data['gps'] = data['gps'].apply(gowalla_to_gps)
        data = data.rename(columns={'userid': 'uid', 'placeid': 'pid', 'datetime': 'time'})
        data = data[['uid', 'pid', 'time', 'gps']]
    else:
        data = None
    return data


def load_data(dataset, min_count=10, timedelta=2, region_size=6):
    """
    load data
    :param dataset: ['meituan', 'foursquare', 'gowalla']
    :param min_count: user visited pois' number minimum value
    :param timedelta: split time by timedelta
    :param region_size: split space, geohash length
    """
    file = './dataset/'+dataset+'/new_data.csv'
    print(file)
    if os.path.isfile(file):
        print('read from file')
        date_col = ['time']
        data = pd.read_csv(file, sep='\t', parse_dates=date_col, infer_datetime_format=True)
        return data
    data = read_dataset(dataset)
    data = data.groupby(['uid']).filter(lambda x: len(x) >= min_count)
    data['hour'] = data['time'].apply(lambda x: x.hour)
    arange = np.arange(0, 25, timedelta)
    data['interval'] = pd.cut(data['hour'], arange, right=False)
    data['region'] = data['gps'].apply(lambda x: geohash.encode(*x)[:region_size])
    if dataset == 'meituan':
        data['user_region'] = data['user_gps'].apply(lambda x: geohash.encode(*x)[:region_size])
        data.to_csv('./dataset/'+dataset+'/new_data.csv', sep='\t', index=0,
                    columns=['uid', 'pid', 'interval', 'region', 'user_region', 'time'])
    else:
        data.to_csv('./dataset/' + dataset + '/new_data.csv', sep='\t', index=0,
                    columns=['uid', 'pid', 'interval', 'region', 'time'])
    return data


if __name__ == '__main__':
    data = read_dataset('gowalla')
    print(data)
