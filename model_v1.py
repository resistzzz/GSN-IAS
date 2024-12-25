

import torch
import torch.nn as nn
from aggregator import *


class NRPOP(nn.Module):
    def __init__(self, opt, n_items, adj_items, anchors, device):
        super(NRPOP, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hidden_size
        self.routing_iter = opt.routing_iter
        self.K = opt.anchor_num
        self.device = device

        self.adj_items = torch.LongTensor(adj_items).to(device)
        self.anchors = torch.LongTensor(anchors).to(device)

        # self.prob = torch.FloatTensor(prob).to(device)
        # self.prob_emb = nn.Parameter(self.prob, requires_grad=True)
        # self.prob_emb = torch.FloatTensor(prob).to(device)

        self.hop = opt.hop
        self.sample_num = opt.sample_num

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        nn.init.xavier_normal_(self.item_embeddings.weight[1:])
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True, bidirectional=False)

        # Global Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = NeighborRoutingAgg(self.hidden_size, self.routing_iter, self.device)
            self.add_module('agg_gnn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # self.w1_attn = nn.Linear(self.hidden_size, self.hidden_size)
        # self.w2_attn = nn.Linear(self.hidden_size, self.hidden_size)
        # self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)
        # nn.init.xavier_normal_(self.w1_attn.weight)
        # nn.init.xavier_normal_(self.w2_attn.weight)
        # nn.init.xavier_normal_(self.v_attn.weight)
        #
        # self.w3_attn = nn.Linear(self.hidden_size, self.hidden_size)
        # self.w4_attn = nn.Linear(self.hidden_size, self.hidden_size)
        # self.u_attn = nn.Linear(self.hidden_size, 1, bias=False)
        # nn.init.xavier_normal_(self.w3_attn.weight)
        # nn.init.xavier_normal_(self.w4_attn.weight)
        # nn.init.xavier_normal_(self.u_attn.weight)

        self.trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.K)
        )
        nn.init.xavier_normal_(self.trans[0].weight)
        nn.init.xavier_normal_(self.trans[2].weight)

        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.a3 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.loss_function1 = nn.CrossEntropyLoss()
        self.loss_function2 = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.LN3 = nn.LayerNorm(self.hidden_size)
        self.LN4 = nn.LayerNorm(self.hidden_size)

        # nn.init.normal_(self.item_embeddings.weight[1:], mean=0, std=1.0/np.sqrt(self.hidden_size))
        # nn.init.normal_(self.cls_embeddings.weight, mean=0, std=1.0/np.sqrt(self.hidden_size))

    def init_h0(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_sess, lengths, mask_inf, mask_1, tau):
        batch_size = inp_sess.size(0)
        seqs_len = inp_sess.size(1)

        x = self.item_embeddings.weight[1:]
        x_nb = self.adj_items[1:]

        out_vectors = [x]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(x=out_vectors[i], x_nb=x_nb)
            out_vectors.append(x)

        item_vectors = out_vectors[0]
        for i in range(1, len(out_vectors)):
            item_vectors = item_vectors + out_vectors[i]
        item_vectors = self.LN1(item_vectors)       # (N - 1) * d

        item_pop_prob = torch.softmax(self.trans(item_vectors) / tau, dim=1)    # N-1 * K
        anchors_emb = item_vectors[self.anchors - 1]  # K * d
        item_pop_vec = torch.mm(item_pop_prob, anchors_emb)     # N-1 * d
        item_pop_vec = self.LN2(item_pop_vec)

        # pad 0 item
        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)
        item_pop_vec = torch.cat((pad_vector, item_pop_vec), dim=0)

        inp_emb = self.dropout(item_vectors[inp_sess])          # bs * L * d
        inp_pop_emb = self.dropout(item_pop_vec[inp_sess])      # bs * L * d
        # inp_vec = torch.cat((inp_emb, inp_pop_emb), dim=-1)     # bs * L * 2d

        h0 = self.init_h0(batch_size)
        H, _ = self.gru(inp_emb, h0)
        H = self.LN3(H)
        ht = H[torch.arange(H.size(0)), lengths - 1]            # bs * d

        h0 = self.init_h0(batch_size)
        H_pop, _ = self.gru(inp_pop_emb, h0)
        H_pop = self.LN4(H_pop)
        ht_pop = H_pop[torch.arange(H_pop.size(0)), lengths - 1]            # bs * d

        # alpha = torch.sigmoid(self.w1_attn(H) + self.w2_attn(ht).unsqueeze(1))
        # alpha = self.v_attn(alpha).squeeze()      # bs * L
        # alpha = torch.softmax(alpha + mask_inf, dim=1).unsqueeze(-1)    # bs * L * 1
        # sess_rep = torch.sum(alpha * H, dim=1)                  # bs * d
        #
        # beta = torch.sigmoid(self.w3_attn(H_pop) + self.w4_attn(ht_pop).unsqueeze(1))
        # beta = self.u_attn(beta).squeeze()      # bs * L
        # beta = torch.softmax(beta + mask_inf, dim=1).unsqueeze(-1)    # bs * L * 1
        # sess_rep_pop = torch.sum(beta * H_pop, dim=1)                  # bs * d

        # scores_item = torch.mm(sess_rep, item_vectors[1:].transpose(1, 0))
        # scores_pop = torch.mm(sess_rep_pop, item_pop_vec[1:].transpose(1, 0))
        scores_item = torch.mm(ht, item_vectors[1:].transpose(1, 0))
        scores_pop = torch.mm(ht_pop, item_pop_vec[1:].transpose(1, 0))

        scores = torch.sigmoid(self.a1) * scores_item + torch.sigmoid(self.a2) * scores_pop

        return scores, scores_item, scores_pop


class NRPOPSess(nn.Module):
    def __init__(self, opt, n_items, adj_items, anchors, device):
        super(NRPOPSess, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hidden_size
        self.routing_iter = opt.routing_iter
        self.K = opt.anchor_num
        self.device = device

        self.adj_items = torch.LongTensor(adj_items).to(device)
        self.anchors = torch.LongTensor(anchors).to(device)

        self.hop = opt.hop
        self.sample_num = opt.sample_num

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        nn.init.xavier_normal_(self.item_embeddings.weight[1:])

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True, bidirectional=False)

        # Global Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = NeighborRoutingAgg(self.hidden_size, self.routing_iter, self.device)
            self.add_module('agg_gnn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # # GAT Agg
        # self.global_agg = []
        # for i in range(self.hop):
        #     agg = GATAgg(self.hidden_size, self.device)
        #     self.add_module('agg_gat_{}'.format(i), agg)
        #     self.global_agg.append(agg)

        self.trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.K)
        )
        nn.init.xavier_normal_(self.trans[0].weight)
        nn.init.xavier_normal_(self.trans[2].weight)

        self.w0 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        nn.init.xavier_normal_(self.w0[0].weight)

        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.loss_function1 = nn.CrossEntropyLoss()
        self.loss_function2 = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.LN3 = nn.LayerNorm(self.hidden_size)
        self.LN4 = nn.LayerNorm(self.hidden_size)

    def init_h0(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_sess, lengths, mask_inf, mask_1, tau):
        batch_size = inp_sess.size(0)
        seqs_len = inp_sess.size(1)

        x = self.item_embeddings.weight[1:]
        x_nb = self.adj_items[1:]

        out_vectors = [x]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(x=out_vectors[i], x_nb=x_nb)
            out_vectors.append(x)

        item_vectors_ = out_vectors[0]
        for i in range(1, len(out_vectors)):
            item_vectors_ = item_vectors_ + out_vectors[i]
        item_vectors = self.LN1(item_vectors_)

        # pad 0 item
        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)

        inp_emb = self.dropout(item_vectors[inp_sess])  # bs * L * d
        inp_prob = torch.softmax(self.trans(inp_emb), dim=-1)  # bs * L * K

        anchors_emb = self.w0(item_vectors[self.anchors])  # K * d
        inp_pop = torch.matmul(inp_prob, anchors_emb)  # bs * L * d

        h0 = self.init_h0(batch_size)
        H, _ = self.gru(inp_emb, h0)
        H = self.LN2(H)
        ht = H[torch.arange(H.size(0)), lengths - 1]

        H_pop, _ = self.gru(inp_pop, h0)
        H_pop = self.LN4(H_pop)
        ht_pop = H_pop[torch.arange(batch_size), lengths - 1]

        scores1 = torch.matmul(ht, item_vectors[1:].transpose(1, 0))
        item_pop_distributions = torch.softmax(self.trans(item_vectors[1:]), dim=1)
        item_pop_vectors_ = torch.matmul(item_pop_distributions, anchors_emb)
        item_pop_vectors = self.LN3(item_pop_vectors_)
        scores2 = torch.matmul(ht_pop, item_pop_vectors.transpose(1, 0))

        scores = F.sigmoid(self.a1) * scores1 + F.sigmoid(self.a2) * scores2
        return scores, scores1, scores2
        # return scores, scores1, scores2, item_vectors_, item_pop_vectors_
        # return scores, scores1, scores2, item_vectors_, item_pop_vectors_, item_pop_distributions


class NRPOPSessAttn(nn.Module):
    def __init__(self, opt, n_items, adj_items, anchors, max_length, device):
        super(NRPOPSessAttn, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hidden_size
        self.routing_iter = opt.routing_iter
        self.K = opt.anchor_num
        self.device = device

        self.adj_items = torch.LongTensor(adj_items).to(device)
        self.anchors = torch.LongTensor(anchors).to(device)

        self.hop = opt.hop
        self.sample_num = opt.sample_num
        self.max_length = max_length

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        nn.init.xavier_normal_(self.item_embeddings.weight[1:])

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True, bidirectional=False)

        # # Global Agg
        # self.global_agg = []
        # for i in range(self.hop):
        #     agg = NeighborRoutingAgg(self.hidden_size, self.routing_iter, self.device)
        #     self.add_module('agg_gnn_{}'.format(i), agg)
        #     self.global_agg.append(agg)

        # GAT Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = GATAgg(self.hidden_size, self.device)
            self.add_module('agg_gnn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.K)
        )
        nn.init.xavier_normal_(self.trans[0].weight)
        nn.init.xavier_normal_(self.trans[2].weight)

        self.w0 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        nn.init.xavier_normal_(self.w0[0].weight)

        # self.position_embeddings = nn.Embedding(self.max_length, self.hidden_size)
        # nn.init.xavier_normal_(self.position_embeddings.weight)
        # Attn
        self.w1_attn1 = nn.Linear(self.hidden_size, self.hidden_size)    # 给h0的
        self.w2_attn1 = nn.Linear(self.hidden_size, self.hidden_size)    # 给ht的
        self.w3_attn1 = nn.Linear(self.hidden_size, self.hidden_size)    # 给h_avg的
        self.w4_attn1 = nn.Linear(self.hidden_size, self.hidden_size)    # 给hi的
        self.q1 = nn.Linear(self.hidden_size, 1, bias=False)
        nn.init.xavier_normal_(self.w1_attn1.weight)
        nn.init.xavier_normal_(self.w2_attn1.weight)
        nn.init.xavier_normal_(self.w3_attn1.weight)
        nn.init.xavier_normal_(self.w4_attn1.weight)
        nn.init.xavier_normal_(self.q1.weight)

        self.w1_attn2 = nn.Linear(self.hidden_size, self.hidden_size)  # 给h0的
        self.w2_attn2 = nn.Linear(self.hidden_size, self.hidden_size)  # 给ht的
        self.w3_attn2 = nn.Linear(self.hidden_size, self.hidden_size)  # 给h_avg的
        self.w4_attn2 = nn.Linear(self.hidden_size, self.hidden_size)  # 给hi的
        self.q2 = nn.Linear(self.hidden_size, 1, bias=False)
        nn.init.xavier_normal_(self.w1_attn2.weight)
        nn.init.xavier_normal_(self.w2_attn2.weight)
        nn.init.xavier_normal_(self.w3_attn2.weight)
        nn.init.xavier_normal_(self.w4_attn2.weight)
        nn.init.xavier_normal_(self.q2.weight)

        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.loss_function1 = nn.CrossEntropyLoss()
        self.loss_function2 = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.LN3 = nn.LayerNorm(self.hidden_size)
        self.LN4 = nn.LayerNorm(self.hidden_size)

    def init_h0(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_sess, lengths, mask_inf, mask_1, tau):
        batch_size = inp_sess.size(0)
        seqs_len = inp_sess.size(1)

        x = self.item_embeddings.weight[1:]
        x_nb = self.adj_items[1:]

        out_vectors = [x]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(x=out_vectors[i], x_nb=x_nb)
            out_vectors.append(x)

        item_vectors_ = out_vectors[0]
        for i in range(1, len(out_vectors)):
            item_vectors_ = item_vectors_ + out_vectors[i]
        item_vectors = self.LN1(item_vectors_)

        # pad 0 item
        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)

        inp_emb = self.dropout(item_vectors[inp_sess])  # bs * L * d

        inp_prob = torch.softmax(self.trans(inp_emb), dim=-1)  # bs * L * K
        anchors_emb = self.w0(item_vectors[self.anchors])  # K * d
        inp_pop = torch.matmul(inp_prob, anchors_emb)  # bs * L * d

        h0 = self.init_h0(batch_size)
        H, _ = self.gru(inp_emb, h0)
        H = self.LN2(H)
        # ht = H[torch.arange(H.size(0)), lengths - 1]

        H_pop, _ = self.gru(inp_pop, h0)
        H_pop = self.LN4(H_pop)
        # ht_pop = H_pop[torch.arange(batch_size), lengths - 1]

        # 第1个attn
        z0_emb = H[torch.arange(batch_size), 0]
        zt_emb = H[torch.arange(batch_size), lengths - 1]
        z_avg1 = (H * mask_1.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        alpha1 = self.w1_attn1(z0_emb) + self.w2_attn1(zt_emb) + self.w3_attn1(z_avg1)
        alpha1 = self.w4_attn1(H) + alpha1.unsqueeze(1)
        alpha1 = self.q1(F.leaky_relu(alpha1)).squeeze(-1)
        alpha1 = torch.softmax(alpha1 + mask_inf, dim=1)
        ht = torch.sum(alpha1.unsqueeze(-1) * H, dim=1)

        # 第2个attn
        z0_pop = H_pop[torch.arange(batch_size), 0]
        zt_pop = H_pop[torch.arange(batch_size), lengths - 1]
        z_avg2 = (H_pop * mask_1.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        alpha2 = self.w1_attn2(z0_pop) + self.w2_attn2(zt_pop) + self.w3_attn2(z_avg2)
        alpha2 = self.w4_attn2(H_pop) + alpha2.unsqueeze(1)
        alpha2 = self.q2(F.leaky_relu(alpha2)).squeeze(-1)
        alpha2 = torch.softmax(alpha2 + mask_inf, dim=1)
        ht_pop = torch.sum(alpha2.unsqueeze(-1) * H_pop, dim=1)

        scores1 = torch.matmul(ht, item_vectors[1:].transpose(1, 0))
        item_pop_distributions = torch.softmax(self.trans(item_vectors[1:]), dim=1)
        item_pop_vectors_ = torch.matmul(item_pop_distributions, anchors_emb)
        item_pop_vectors = self.LN3(item_pop_vectors_)
        scores2 = torch.matmul(ht_pop, item_pop_vectors.transpose(1, 0))

        scores = F.sigmoid(self.a1) * scores1 + F.sigmoid(self.a2) * scores2
        return scores, scores1, scores2


class LightGCNLC(nn.Module):
    def __init__(self, opt, n_items, anchors, A, device):
        super(LightGCNLC, self).__init__()
        self.n_items = n_items
        self.hidden_size = opt.hidden_size
        self.K = opt.anchor_num
        self.device = device

        self.anchors = torch.LongTensor(anchors).to(device)

        self.hop = opt.hop
        self.sample_num = opt.sample_num

        self.A = A.to(self.device)

        # # GCN Agg
        # self.global_agg = []
        # for i in range(self.hop):
        #     agg = GCNAgg(self.hidden_size)
        #     self.add_module('agg_gcn_{}'.format(i), agg)
        #     self.global_agg.append(agg)

        # LightGCN Agg
        self.global_agg = []
        for i in range(self.hop):
            agg = LightGCNAgg(self.hidden_size)
            self.add_module('agg_lightgcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        nn.init.xavier_normal_(self.item_embeddings.weight)

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True, bidirectional=False)

        self.trans = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.K)
        )
        nn.init.xavier_normal_(self.trans[0].weight)
        nn.init.xavier_normal_(self.trans[2].weight)

        # self.w0 = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size)
        # )
        # nn.init.xavier_normal_(self.w0[0].weight)
        # nn.init.xavier_normal_(self.w0[2].weight)
        self.w0 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        nn.init.xavier_normal_(self.w0[0].weight)

        self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.loss_function1 = nn.CrossEntropyLoss()
        self.loss_function2 = nn.CrossEntropyLoss()
        self.loss_function = nn.CrossEntropyLoss()

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.LN3 = nn.LayerNorm(self.hidden_size)
        self.LN4 = nn.LayerNorm(self.hidden_size)

    def init_h0(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, inp_sess, lengths, mask_inf, mask_1, tau):
        batch_size = inp_sess.size(0)
        seqs_len = inp_sess.size(1)

        x = self.item_embeddings.weight[1:]

        out_vectors = [x]
        for i in range(self.hop):
            aggregator = self.global_agg[i]
            x = aggregator(A=self.A, x=out_vectors[i])
            out_vectors.append(x)

        item_vectors = out_vectors[0]
        for i in range(1, len(out_vectors)):
            item_vectors = item_vectors + out_vectors[i]
        item_vectors = self.LN1(item_vectors)

        # pad 0 item
        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)

        inp_emb = self.dropout(item_vectors[inp_sess])  # bs * L * d
        inp_prob = torch.softmax(self.trans(inp_emb), dim=-1)  # bs * L * K

        anchors_emb = self.w0(item_vectors[self.anchors])  # K * d
        inp_pop = torch.matmul(inp_prob, anchors_emb)  # bs * L * d

        h0 = self.init_h0(batch_size)
        H, _ = self.gru(inp_emb, h0)
        H = self.LN2(H)
        ht = H[torch.arange(H.size(0)), lengths - 1]

        H_pop, _ = self.gru(inp_pop, h0)
        H_pop = self.LN4(H_pop)
        ht_pop = H_pop[torch.arange(batch_size), lengths - 1]

        scores1 = torch.matmul(ht, item_vectors[1:].transpose(1, 0))
        item_pop_distributions = torch.softmax(self.trans(item_vectors[1:]), dim=1)
        item_pop_vectors = self.LN3(torch.matmul(item_pop_distributions, anchors_emb))
        # item_pop_vectors = torch.matmul(item_pop_distributions, anchors_emb)
        scores2 = torch.matmul(ht_pop, item_pop_vectors.transpose(1, 0))

        scores = F.sigmoid(self.a1) * scores1 + F.sigmoid(self.a2) * scores2
        return scores, scores1, scores2
