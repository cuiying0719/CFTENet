import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from layers import UnionRGCNLayer, RGCNBlockLayer, UnionRGCNLayer_His
from model import BaseRGCN
from decoder import *


# class RGCNCell(BaseRGCN):
#     def build_hidden_layer(self, idx):
#         act = F.rrelu
#         if idx:
#             self.num_basis = 0
#         print("activate function: {}".format(act))
#         if self.skip_connect:
#             sc = False if idx == 0 else True
#         else:
#             sc = False
#         if self.encoder_name == "convgcn":
#             return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                              activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
#         else:
#             raise NotImplementedError


#     def forward(self, g, init_ent_emb, init_rel_emb):
#         if self.encoder_name == "convgcn":
#             node_id = g.ndata['id'].squeeze()
#             g.ndata['h'] = init_ent_emb[node_id]
#             x, r = init_ent_emb, init_rel_emb
#             for i, layer in enumerate(self.layers):
#                 layer(g, [], r[i])
#             return g.ndata.pop('h')
#         else:
#             if self.features is not None:
#                 print("----------------Feature is not None, Attention ------------")
#                 g.ndata['id'] = self.features
#             node_id = g.ndata['id'].squeeze()
#             g.ndata['h'] = init_ent_emb[node_id]
#             if self.skip_connect:
#                 prev_h = []
#                 for layer in self.layers:
#                     prev_h = layer(g, prev_h)
#             else:
#                 for layer in self.layers:
#                     layer(g, [])
#             return g.ndata.pop('h')
        
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb, time_params):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i], time_params)
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')
        
class RGCNCell_His(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer_His(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        # self.rgcn_his = RGCNCell_His(num_ents,
        #                      h_dim,
        #                      h_dim,
        #                      num_rels * 2,
        #                      num_bases,
        #                      num_basis,
        #                      num_hidden_layers,
        #                      dropout,
        #                      self_loop,
        #                      skip_connect,
        #                      encoder_name,
        #                      self.opn,
        #                      self.emb_rel,
        #                      use_cuda,
        #                      analysis)
        
        # self.generate_gate = nn.Linear(self.h_dim * 2, 1)
        # nn.init.xavier_uniform_(self.generate_gate.weight, gain=nn.init.calculate_gain('relu'))

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight , gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        # self.relation_cell_2 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        # self.entity_cell_2 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            model_params = {
            # 'input_size',C_in
            'input_size':   4,
            # 单步，预测未来一个时刻
            'output_size':  1,
            # 'num_channels': [32,32,32,32,32,32,32,32,32],
            # 'num_channels': [32,32,32,32,32],
            # 'num_channels': [50,50,50,50,50,50,50,50,50],
            # 'num_channels': [50,50,50,50,50],
            'num_channels': [16,16,16,16,16],
            # 'num_channels': [8,8,8,8,8],
            'kernel_size':  3,
            'dropout':      0.2
            }

            # self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob1 = TCNDecoder(**model_params)
            self.decoder_ob2 = TCNDecoder(**model_params)
            self.rdecoder_re1 = TCNDecoderR(**model_params)
            self.rdecoder_re2 = TCNDecoderR(**model_params)
        else:
            raise NotImplementedError 

    def forward(self, g_list, input_his, static_graph, use_cuda, time_params):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        # 可以考虑替换成F.normalize(self.dynamic_emb)或者新建一个实体的嵌入矩阵-------------------
        # self.h_his = F.normalize(self.dynamic_emb)#self.h
        # -------------------------------------------------------------------------------

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            # -----------------
            # current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0], time_params)
            # -----------------
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        
        # input_his = input_his.to(self.gpu)
        # temp_e_his = self.h_his[input_his.r_to_e]
        # x_input_his = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
        # for span, r_idx in zip(input_his.r_len, input_his.uniq_r):
        #     x = temp_e_his[span[0]:span[1],:]
        #     x_mean = torch.mean(x, dim=0, keepdim=True)
        #     x_input_his[r_idx] = x_mean
        # # 这里可以考虑直接用self.emb_rel，或者F.normalize(self.emb_rel)，或者新建一个关系的嵌入矩阵-----------
        # # x_input_his = torch.cat((self.emb_rel, x_input_his), dim=1)
        # self.h_0_his = F.normalize(self.emb_rel)#self.relation_cell_2(x_input_his, self.emb_rel)
        # # ----------------------------------------------------------------------
        # # self.h_0_his = F.normalize(self.h_0_his) if self.layer_norm else self.h_0_his
        # current_h_his = self.rgcn_his.forward(input_his, self.h_his, [self.h_0_his, self.h_0_his])
        # current_h_his = F.normalize(current_h_his) if self.layer_norm else current_h_his
        # # 这里是按照上面的写法，可以不加------
        # # current_h_his = self.entity_cell_2(current_h_his, self.h_his)
        # # current_h_his = F.normalize(current_h_his) if self.layer_norm else current_h_his
        # # ---------------------------------

        # return history_embs, current_h_his, static_emb, self.h_0, gate_list, degree_list
        return history_embs, None, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, input_his, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, current_his, _, r_emb, _, _ = self.forward(test_graph, input_his, static_graph, use_cuda, [self.weight_t1, self.bias_t1, self.weight_t2, self.bias_t2, self.time_interval])
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # 合并信息-----------------
            # ent_weight = torch.sigmoid(F.leaky_relu(self.generate_gate(torch.cat([embedding, current_his], -1))))
            # embedding = ent_weight * embedding + (1-ent_weight) * current_his
            # embedding = F.normalize(embedding) if self.layer_norm else embedding
            # ------------------------


            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)
            score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score = torch.log(score)

            return all_triples, score, score_rel


    def get_loss(self, glist, input_his, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, current_his, static_emb, r_emb, _, _ = self.forward(glist, input_his, static_graph, use_cuda, [self.weight_t1, self.bias_t1, self.weight_t2, self.bias_t2, self.time_interval])
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        # 这里的时间嵌入可以考虑放在forward里面，用来作为知识图谱的门控信息----------
        time_embs = self.get_init_time(all_triples)
        # ----------------------------------------------------------------------

        # 合并信息------------------
        # ent_weight = torch.sigmoid(F.leaky_relu(self.generate_gate(torch.cat([pre_emb, current_his], -1))))
        # pre_emb = ent_weight * pre_emb + (1-ent_weight) * current_his
        # pre_emb = F.normalize(pre_emb) if self.layer_norm else pre_emb
        # -------------------------

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding = global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h






