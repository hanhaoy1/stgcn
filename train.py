import numpy as np
import torch
from load_graph import Graph
import torch.nn.functional as F
from model import Recommender, STGCN
import time
import dgl
from functools import partial
from collections import defaultdict
import math

torch.cuda.set_device(0)
device = torch.device('cuda:0')


def edge_func(time0, edges):
    rel_type = edges.data['type']
    time = edges.data['time']
    weight = edges.data['weight']
    norm = edges.data['norm']
    timedelta = torch.abs(time - time0) / 2
    msk = rel_type < 3
    timedelta[msk] = 0
    timedelta = -timedelta.float()
    time_weight = torch.exp(timedelta)
    final_weight = weight * time_weight
    final_weight = final_weight * (1 / norm)
    return {'final_weight': final_weight}


def edge_func1(edges):
    rel_type = edges.data['type']
    time = edges.data['time']
    weight = edges.data['weight']
    norm = edges.data['norm']
    final_weight = weight * (1 / norm)
    return {'final_weight': final_weight}


def recallk(graph, model, dim=64, batch_size=1024, layers=2, samples=5, has_user_region=True):
    k_list = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # full rank recall
    test_record = graph.time_test
    g = graph.g
    g.readonly()
    pid2rid = torch.LongTensor(graph.pid2rid[['p', 'r']].values).cuda()
    pid2indices = {}
    for index, pid in enumerate(graph.pid2rid['p'].to_list()):
        pid2indices[pid] = index
    accuracy = defaultdict(int)
    ndcg = defaultdict(float)
    for t in range(0, 24, 2):
        g.apply_edges(partial(edge_func, t))
        sampler = dgl.contrib.sampling.NeighborSampler(
            g,
            batch_size,
            samples,
            layers,
            seed_nodes=torch.arange(g.number_of_nodes()), 
            transition_prob='final_weight',
            num_workers=16,
        )
        emb = torch.empty((g.number_of_nodes(), dim), device=device)
        for nf in sampler:
            nf.copy_from_parent(ctx=device)
            batch_nids = nf.layer_parent_nid(-1).long()
            h = model.infer(nf)
            emb[batch_nids] = h
        record_t = test_record[t]
        pois_indices = np.array([pid2indices[i] for i in record_t[:, 1]]).reshape(-1, 1)
        record_t = np.concatenate((record_t, pois_indices), 1)
        tests = torch.from_numpy(record_t).cuda()
        # test_batches = tests.split(batch_size)
        for test in tests:
            user = emb[test[0]]
            if has_user_region:
                user_region = emb[test[2]]
                true_indices = test[4]
            else:
                true_indices = test[3]
            pois = emb[pid2rid[:, 0]]
            pois_region = emb[pid2rid[:, 1]]
            if has_user_region:
                scores = user * pois + 0.1 * user * pois_region + user_region * pois + user_region * pois_region
            else:
                scores = user * pois + user * pois_region
            scores = scores.sum(1)
            scores, indices = torch.sort(scores, descending=True)
            position = (indices == true_indices).nonzero().item()
            for k in k_list:
                if position < k:
                    accuracy[k] += 1
                    ndcg[k] += 1 / math.log2(position+2)
    for k in k_list:
        accuracy[k] /= len(graph.test)
        ndcg[k] /= len(graph.test)
    print(accuracy)
    print(ndcg)


def main(dataset):
    batch_size = 1024
    graph = Graph(dataset)
    if dataset == 'meituan':
        data_size = 6
        has_user_region=True
    else:
        data_size = 5
        has_user_region = False
    g = graph.g
    g.readonly()
    embeddings = graph.embeddings
    num_nodes = graph.g.number_of_nodes()
    model = Recommender(STGCN(num_nodes, 64, 27, 2, None, embeddings))
    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    model.train()
    print('train neighbors')
    for i in range(30):
        total_loss = 0
        g.apply_edges(edge_func1)
        neighbors = graph.neighbors
        neg_neighbors = np.random.choice(range(num_nodes), neighbors.shape[0], replace=True).reshape(-1, 1)
        neighbors_data = np.concatenate((neighbors, neg_neighbors), axis=1)
        data = torch.from_numpy(neighbors_data).cuda()
        seed_nodes = data.reshape(-1)
        batches = data.split(batch_size)
        sampler = dgl.contrib.sampling.NeighborSampler(
            g,
            batch_size * 3,
            5,
            2,
            seed_nodes=seed_nodes,
            num_workers=11)
        count = 0
        for batch, nf in zip(batches, sampler):
            nf.copy_from_parent(ctx=device)
            batch_nid = nf.map_from_parent_nid(-1, batch.reshape(-1), True)
            batch_nid = batch_nid.reshape(-1, 3).cuda()
            loss = model.train_region(nf, batch_nid)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            count += 1
        print('loss', total_loss / count)

    for epoch in range(300):
        model.train()
        begin = time.time()
        total_loss = 0
        count = 0
        for t in range(0, 24, 2):
            g.apply_edges(partial(edge_func, t))
            pos = graph.time_train[t]
            neg_pois = graph.pid2rid[['p', 'r']].sample(n=pos.shape[0], replace=True).to_numpy(copy=True)
            data = np.concatenate((pos, neg_pois), axis=1)
            data.astype(np.int)
            data = torch.from_numpy(data).cuda()
            seed_nodes = data.reshape(-1)
            batches = data.split(batch_size)
            sampler = dgl.contrib.sampling.NeighborSampler(
                g,
                batch_size * data_size,
                5,
                2,
                seed_nodes=seed_nodes,
                transition_prob='final_weight',
                prefetch=False,
                num_workers=11)
            for batch, nf in zip(batches, sampler):
                nf.copy_from_parent(ctx=device)
                batch_nid = nf.map_from_parent_nid(-1, batch.reshape(-1), True)
                batch_nid = batch_nid.reshape(-1, data_size).cuda()
                loss = model(nf, batch_nid, has_user_region=has_user_region)
                opt.zero_grad()
                loss.backward()
                total_loss += loss.item()
                opt.step()
                count += 1
        print('epoch:{}, loss:{}, time:{}'.format(epoch, total_loss / count, time.time() - begin))
        if epoch % 20 ==0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                recallk(graph, model, has_user_region=has_user_region)
    model.eval()
    with torch.no_grad():
        recallk(graph, model, has_user_region=has_user_region)


if __name__ == "__main__":
    dataset = 'meituan'
    main(dataset)


