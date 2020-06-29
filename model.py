import torch
import torch.nn as nn

torch.cuda.set_device(0)
device = torch.device('cuda:0')


class Layer(nn.Module):
    def __init__(self, dim, num_rels, activation):
        super(Layer, self).__init__()
        self.dim = dim
        self.num_rels = num_rels
        self.weight = nn.Parameter(torch.Tensor(num_rels, dim, dim))
        # self.W = nn.Parameter(torch.Tensor(dim, dim))
        # nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def message_func(self, edges):
        h = edges.src['h']
        rel_type = edges.data['type']
        weight = edges.data['final_weight']
        h *= weight.unsqueeze(1)
        w = self.weight[rel_type]
        msg = torch.bmm(h.unsqueeze(1), w).squeeze()
        return {'msg': msg}

    def reduce_func(self, nodes):
        h = nodes.data['h']
        # h = torch.matmul(h, )
        # h = self.W(h)
        m = nodes.mailbox['msg']
        m = m.sum(dim=1, keepdim=True)
        m = m / m.norm(dim=2, keepdim=True).clamp(min=1e-6)
        h = h.unsqueeze(1)
        h_new = torch.cat((m, h), 1).sum(dim=1)
        if self.activation:
            h_new = self.activation(h_new)
        return {'h': h_new / h_new.norm(dim=1, keepdim=True)}

    def forward(self, nf, i_layer):
        nf.block_compute(i_layer, self.message_func, self.reduce_func)
        return nf


class STGCN(nn.Module):
    def __init__(self, num_nodes, dim, num_rels, num_layers, activation, embeddings):
        super(STGCN, self).__init__()
        self.dim = dim
        self.num_rels = num_rels
        self.activation = activation
        self.embeddings = nn.Embedding(num_nodes, dim)
        # nn.init.xavier_uniform_(self.embeddings.weight)
        self.embeddings.weight.data = embeddings
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Layer(dim, num_rels, activation))

    def forward(self, nf):
        for i in range(nf.num_layers):
            nids = nf.layer_parent_nid(i).cuda()
            nf.layers[i].data['h'] = self.embeddings(nids)
        for i in range(self.num_layers):
            nf = self.layers[i](nf, i)
        result = nf.layers[self.num_layers].data['h']
        return result


class Recommender(nn.ModuleList):
    """
    Recommender
    score = Ut * P + Ut * Lp + P * Lu + Lp * Lu
    """
    def __init__(self, gcn):
        super(Recommender, self).__init__()
        self.gcn = gcn
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, nf, data, has_user_region=True):
        h = self.gcn(nf)
        # already convert to nodeflow id
        user = h[data[:, 0]]
        pos_poi = h[data[:, 1]]
        if has_user_region:
            user_region = h[data[:, 2]]
            poi_region = h[data[:, 3]]
            neg_poi = h[data[:, 4]]
            neg_poi_region = h[data[:, 5]]
            pos_score = user * pos_poi + 0.1 * user * poi_region + pos_poi * user_region + poi_region * user_region
            neg_score = user * neg_poi + 0.1 * user * neg_poi_region + neg_poi * user_region + neg_poi_region * user_region
        else:
            poi_region = h[data[:, 2]]
            neg_poi = h[data[:, 3]]
            neg_poi_region = h[data[:, 4]]
            pos_score = user * pos_poi + user * poi_region
            neg_score = user * neg_poi + user * neg_poi_region
        pos_score = pos_score.sum(1)
        neg_score = neg_score.sum(1)
        maxi = self.logsigmoid(pos_score - neg_score)
        loss = -maxi.mean()
        return loss

    def train_region(self, nf, data):
        h = self.gcn(nf)
        r1 = h[data[:, 0]]
        r2 = h[data[:, 1]]
        r3 = h[data[:, 2]]
        pos_score = (r1 * r2).sum(1)
        neg_score = (r1 * r3 + r2 * r3).sum(1)
        maxi = self.logsigmoid(pos_score - neg_score)
        loss = -maxi.mean()
        return loss

    def infer(self, nf):
        h = self.gcn(nf)
        return h
