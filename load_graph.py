import numpy as np
import pandas as pd
import os
from data_loader import load_data
import json
import dgl
import torch
import math
from collections import defaultdict

class Graph(object):
    def __init__(self, dataset):
        super(Graph, self).__init__()
        self.dataset = dataset
        self.pid2pid, self.rid2rid, self.pid2rid = self.load_datas()
        self.user2id, self.poi2id, self.region2id = self.load_id_map()
        self.users = set(self.user2id.keys())
        self.pois = set(self.poi2id.keys())
        self.regions = set(self.region2id.keys())
        self.num_users, self.num_pois, self.num_regions = map(len, [self.users, self.pois, self.regions])
        print(self.num_users, self.num_pois)
        self.train, self.test = self.load_train_test()
        self.time_train, self.time_test = self.time_convert()
        self.pid_pid_norm, self.user_pid_norm, self.region_region_norm = self.norm()
        self.g = self.build_graph()
        self.neighbors = self.get_neighbors()
        print('build graph done')
        self.embeddings = self.get_init_embeddings()
        print('load embeddings done')
        # self.time_user_records = self.build_records()

    def load_datas(self):
        poi2poi_file = os.path.join('./dataset/', self.dataset, 'poi2poi_train.txt')
        region_file = os.path.join('./dataset/', self.dataset, 'region2region.txt')
        poi2poi = pd.read_csv(poi2poi_file, sep='\t', header=None, names=['p1', 'p2', 'w'])
        # poi2poi = poi2poi[poi2poi['w'] > 1].reset_index()
        region2region = pd.read_csv(region_file, sep='\t', header=None, names=['r1', 'r2', 'w'])
        poi2region_file = os.path.join('./dataset/', self.dataset, 'poi_region.txt')
        poi2region = pd.read_csv(poi2region_file, sep='\t', header=None, names=['p', 'r', 'w'])
        return poi2poi, region2region, poi2region

    def load_id_map(self):
        user2id_file = os.path.join('./dataset', self.dataset, 'user2id.json')
        poi2id_file = os.path.join('./dataset', self.dataset, 'poi2id.json')
        region2id_file = os.path.join('./dataset', self.dataset, 'region2id.json')
        user2id = json.load(open(user2id_file))
        poi2id = json.load(open(poi2id_file))
        region2id = json.load(open(region2id_file))
        return user2id, poi2id, region2id

    def load_train_test(self):
        train_file = os.path.join('./dataset', self.dataset, 'train.csv')
        test_file = os.path.join('./dataset', self.dataset, 'test_new.csv')
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        train['time'] = train['interval'].apply(lambda x: int(x.split(',')[0][1:]))
        test['time'] = test['interval'].apply(lambda x: int(x.split(',')[0][1:]))
        return train, test

    def time_convert(self):
        time_train = {}
        time_test = {}
        train_grouped = self.train.groupby(['time'])
        test_grouped = self.test.groupby(['time'])
        if self.dataset == 'meituan':
            for time, group in train_grouped:
                time_train[time] = group[['uid', 'pid', 'user_region', 'region']].to_numpy(copy=True)
            for time, group in test_grouped:
                time_test[time] = group[['uid', 'pid', 'user_region', 'region']].to_numpy(copy=True)
        else:
            for time, group in train_grouped:
                time_train[time] = group[['uid', 'pid', 'region']].to_numpy(copy=True)
            for time, group in test_grouped:
                time_test[time] = group[['uid', 'pid',  'region']].to_numpy(copy=True)
        return time_train, time_test

    def norm(self):
        '''calculate the norm of every type of edges'''
        pid_pid_norm = defaultdict(int)
        for index, row in self.pid2pid.iterrows():
            p1 = row['p1']
            p2 = row['p2']
            w = row['w']
            # pid_pid_norm[p1] += w
            # pid_pid_norm[p2] += w
            pid_pid_norm[p1] += 1
            pid_pid_norm[p2] += 1
        user_pid_norm = defaultdict(int)
        for index, row in self.train.iterrows():
            user = row['uid']
            poi = row['pid']
            user_pid_norm[user] += 1
            user_pid_norm[poi] += 1
        region_region_norm = defaultdict(int)
        for index, row in self.rid2rid.iterrows():
            r1 = row['r1']
            r2 = row['r2']
            region_region_norm[r1] += 1
            region_region_norm[r2] += 1
        return pid_pid_norm, user_pid_norm, region_region_norm

    def read_vector(self, file):
        vec_dict = {}
        with open(file) as f:
            info = f.readline()
            count, dim = info.strip().split()
            # assert count == self.num_users + self.num_pois + self.num_regions
            print('vec', count)
            print('total', self.num_users + self.num_pois + self.num_regions)
            for line in f:
                info = line.strip().split()
                key = info[0]
                value = torch.tensor([float(i) for i in info[1:]])
                vec_dict[key] = value
        return vec_dict

    def get_init_embeddings(self):
        vec_dict = self.read_vector('./dataset/' + self.dataset + '/line_embedding.txt')
        embeddings = torch.zeros((self.num_users+self.num_pois+self.num_regions, 64))
        torch.nn.init.xavier_uniform_(embeddings)
        for k, v in vec_dict.items():
            k = int(k)
            embeddings[k] = v
        return embeddings

    def get_neighbors(self):
        pid2pid = self.pid2pid[['p1', 'p2']].values
        rid2rid = self.rid2rid[['r1', 'r2']].values
        pid2rid = self.pid2rid[['p', 'r']].values
        uid2pid = self.train[['uid', 'pid']].values
        neighbors = np.concatenate((pid2pid, rid2rid, pid2rid, uid2pid), axis=0)
        return neighbors

    def build_graph(self):
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(self.num_users + self.num_pois + self.num_regions)
        # add poi to poi edges
        g.add_edges(
            self.pid2pid['p1'],
            self.pid2pid['p2'],
            data={'weight': torch.FloatTensor(self.pid2pid['w']),
                  'type': torch.LongTensor([0]*len(self.pid2pid)),
                  'time': torch.IntTensor([-1]*len(self.pid2pid)),
                  'norm': torch.FloatTensor([self.pid_pid_norm[i] for i in self.pid2pid['p2']])
                  }
        )
        g.add_edges(
            self.pid2pid['p2'],
            self.pid2pid['p1'],
            data={'weight': torch.FloatTensor(self.pid2pid['w']),
                  'type': torch.LongTensor([0]*len(self.pid2pid)),
                  'time': torch.IntTensor([-1]*len(self.pid2pid)),
                  'norm': torch.FloatTensor([self.pid_pid_norm[i] for i in self.pid2pid['p1']])
                  }
        )
        # add region to region edges
        g.add_edges(
            self.rid2rid['r1'],
            self.rid2rid['r2'],
            data={'weight': torch.FloatTensor([1] * len(self.rid2rid)),
                  'type': torch.LongTensor([1]*len(self.rid2rid)),
                  'time': torch.IntTensor([-1]*len(self.rid2rid)),
                  'norm': torch.FloatTensor([self.region_region_norm[i] for i in self.rid2rid['r2']])
                  }
        )
        g.add_edges(
            self.rid2rid['r2'],
            self.rid2rid['r1'],
            data={'weight': torch.FloatTensor([1] * len(self.rid2rid)),
                  'type': torch.LongTensor([1]*len(self.rid2rid)),
                  'time': torch.IntTensor([-1]*len(self.rid2rid)),
                  'norm': torch.FloatTensor([self.region_region_norm[i] for i in self.rid2rid['r1']])
                  }
        )
        # add region to poi edges
        g.add_edges(
            self.pid2rid['r'],
            self.pid2rid['p'],
            data={'weight': torch.FloatTensor([1] * len(self.pid2rid)),
                  'type': torch.LongTensor([2] * len(self.pid2rid)),
                  'time': torch.IntTensor([-1]*len(self.pid2rid)),
                  'norm': torch.FloatTensor([1] * len(self.pid2rid))
                  }
        )
        # add region to user edges
        # data1 = self.train[['uid', 'region', 'time']]
        # data1 = data1.groupby(data1.columns.tolist()).size().reset_index().rename(columns={0: 'weight'})
        # data1['type'] = data1['time'].apply(lambda x: 27 + x // 2)
        # g.add_edges(
        #     data1['region'],
        #     data1['uid'],
        #     data={'weight': torch.FloatTensor(data1['weight']),
        #           'type': torch.LongTensor(data1['type']),
        #           'time': torch.IntTensor(data1['time']),
        #           'norm': torch.FloatTensor([self.user_pid_norm[i] for i in data1['uid']])
        #           }
        # )

        # add user to poi edges
        data = self.train[['uid', 'pid', 'time']]
        data = data.groupby(data.columns.tolist()).size().reset_index().rename(columns={0: 'weight'})
        data['type'] = data['time'].apply(lambda x: 3 + x // 2)
        data['type1'] = data['time'].apply(lambda x: 15 + x // 2)
        g.add_edges(
            data['uid'],
            data['pid'],
            data={'weight': torch.FloatTensor(data['weight']),
                  'type': torch.LongTensor(data['type']),
                  'time': torch.IntTensor(data['time']),
                  'norm': torch.FloatTensor([self.user_pid_norm[i] for i in data['pid']])
                  }
        )
        # add poi to user edges
        g.add_edges(
            data['pid'],
            data['uid'],
            data={'weight': torch.FloatTensor(data['weight']),
                  'type': torch.LongTensor(data['type1']),
                  'time': torch.IntTensor(data['time']),
                  'norm': torch.FloatTensor([self.user_pid_norm[i] for i in data['uid']])
                  }
        )
        return g


if __name__ == '__main__':
    graph = Graph(dataset='meituan')
    # print(graph.region_region_norm)
    print(graph.num_users, graph.num_pois, graph.num_regions)
    print(len(graph.train), len(graph.test))
    # rid2rid = graph.rid2rid.values
    # print(graph.region2id.values())
    # neg_rid = np.random.choice(list(graph.region2id.values()), rid2rid.shape[0], replace=True).reshape(-1, 1)
    # rid2rid = np.concatenate((rid2rid, neg_rid), axis=1)
    # print(rid2rid)
    # g = graph.g
    # print(g)
