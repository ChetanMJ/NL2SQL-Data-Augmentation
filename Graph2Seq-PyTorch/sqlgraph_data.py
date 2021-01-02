import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import *
from collections import OrderedDict
import json
from preprocess import word_tokenize
import config as conf

class SQLGraph_Dataset(Dataset):
    def __init__(self, input_path, word2idx):
        self.word2idx = word2idx
        self.seqs = [] # questions
        # self.idx_seqs = []
        self.g_ids = []
        self.g_ids_features = []
        self.g_adj = []
        self.sql_seqs = [] # source sql sequence
        # self.idx_sql_seqs = []

        with open(input_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                jo = json.loads(line, object_pairs_hook=OrderedDict)
                if len(jo['sql']) == 0:
                    continue
                if 'text_tokens' in jo: # simple workaround for Spider dataset
                    seq = jo['text_tokens'] + ['<eos>']
                else:
                    seq = word_tokenize(jo['text']) + ['<eos>']
                # idx_seq = [word2idx[w] if w in word2idx else word2idx['<oov>'] for w in seq]
                self.seqs.append(seq)
                # self.idx_seqs.append(torch.tensor(idx_seq))
                sql_seq = jo['sql']
                # idx_sql_seq = [word2idx[w] if w in word2idx else word2idx['<oov>'] for w in sql_seq]
                self.sql_seqs.append(sql_seq)
                # self.idx_sql_seqs.append(torch.tensor(idx_sql_seq))
                
                self.g_ids.append(jo['g_ids'])
                g_ids_features = jo['g_ids_features']
                for k, v in g_ids_features.items():
                    feat_tokens = word_tokenize(v)
                    g_ids_features[k] = [word2idx[w] if w in word2idx else word2idx['<oov>'] for w in feat_tokens]
                self.g_ids_features.append(jo['g_ids_features'])
                self.g_adj.append(jo['g_adj'])

    def __len__(self):
        return len(self.g_ids)

    def __getitem__(self, index):
        idx_sql_seq, ext_idx_sql_seq, oov_list = self.src2idx(self.sql_seqs[index])
        idx_seq, ext_idx_seq = self.dst2idx(self.seqs[index], oov_list)

        return idx_seq, ext_idx_seq, len(idx_seq), \
            idx_sql_seq, ext_idx_sql_seq, len(idx_sql_seq), \
            self.g_ids[index], self.g_ids_features[index], self.g_adj[index], oov_list

    # from source sentence to ids and extended ids (with OOV words)
    # also return the OOV list
    def src2idx(self, tokens):
        ids = []
        extended_ids = []
        oov_list = []
        for w in tokens:
            if w in self.word2idx:
                ids.append(self.word2idx[w])
                extended_ids.append(self.word2idx[w])
            else:
                ids.append(conf.OOV_IDX)
                if w not in oov_list:
                    oov_list.append(w)
                extended_ids.append(len(self.word2idx) + oov_list.index(w))
        
        return torch.tensor(ids).long(), torch.tensor(extended_ids).long(), oov_list

    # from dest sentence to ids and extended ids
    # vocab: original vocab + OOV word list from src 
    def dst2idx(self, tokens, oov_list):
        ids = []
        extended_ids = []
        for w in tokens:
            if w in self.word2idx:
                ids.append(self.word2idx[w])
                extended_ids.append(self.word2idx[w])
            else:
                ids.append(conf.OOV_IDX)
                if w in oov_list:
                    extended_ids.append(len(self.word2idx) + oov_list.index(w))
                else:
                    extended_ids.append(conf.OOV_IDX) # unknown to the decoder
        
        return torch.tensor(ids).long(), torch.tensor(extended_ids).long()

# Returns:
# idx_seqs: NL question word idx sequence
# batch_g_ids: graph node ids for the batch (0-total number of nodes)
# batch_g_nodes: node ids for each data point (list of list)
# batch_features: node features
# batch_fw/bw_adj: adjacency lists
def collate(batch_data):
    idx_seqs, ext_idx_seqs, seqs_lens, idx_sql_seqs, ext_idx_sql_seqs, sql_seqs_lens, g_ids, g_ids_features, g_adj, oov_list = zip(*batch_data)
    idx_seqs = pad_sequence(idx_seqs, batch_first=True)
    ext_idx_seqs = pad_sequence(ext_idx_seqs, batch_first=True)
    idx_sql_seqs = pad_sequence(idx_sql_seqs, batch_first=True)
    ext_idx_sql_seqs = pad_sequence(ext_idx_sql_seqs, batch_first=True)
    # construct batch graph
    batch_g_ids = []
    batch_g_nodes = []
    batch_fw_adj = []
    batch_bw_adj = []
    for i, ids in enumerate(g_ids):
        offset = len(batch_g_ids)
        nodes = []
        for _, id in ids.items():
            batch_g_ids.append(offset + id)
            nodes.append(offset + id)
            batch_fw_adj.append([])
            batch_bw_adj.append([])
        batch_g_nodes.append(nodes)
        for id, neighbor_ids in g_adj[i].items():
            for nid in neighbor_ids:
                batch_fw_adj[offset + int(id)].append(offset + int(nid))
                batch_bw_adj[offset + int(nid)].append(offset + int(id))
            
    max_degree = 0
    pad_node_id = len(batch_fw_adj)
    for i, nids in enumerate(batch_fw_adj):
        max_degree = max(max_degree, len(nids))
    for i, nids in enumerate(batch_bw_adj):
        max_degree = max(max_degree, len(nids))
    max_degree = min(max_degree, conf.sample_size_per_layer)
    
    for i, nids in enumerate(batch_fw_adj):
        nids.extend([pad_node_id for _ in range(len(nids), max_degree)])
    batch_fw_adj.append([pad_node_id for _ in range(max_degree)])
    batch_fw_adj = torch.tensor(batch_fw_adj).long()
    
    for i, nids in enumerate(batch_bw_adj):
        nids.extend([pad_node_id for _ in range(len(nids), max_degree)])
    batch_bw_adj.append([pad_node_id for _ in range(max_degree)])
    batch_bw_adj = torch.tensor(batch_bw_adj).long()
    
    batch_features = []
    batch_features_lens = []
    for d in g_ids_features:
        for k, v in d.items():
            batch_features.append(torch.tensor(v).long())
            batch_features_lens.append(len(v))
    batch_features = pad_sequence(batch_features, batch_first=True)
    
#     print(batch_g_ids)
#     print(batch_g_nodes)
#     print(batch_features)
#     print(batch_fw_adj)
#     print(batch_bw_adj) # The original code has a bug for bw?
    
    return {
        'idx_seqs': idx_seqs.to(conf.device),
        'ext_idx_seqs': ext_idx_seqs.to(conf.device),
        'seqs_lens': seqs_lens,
        'idx_sql_seqs': idx_sql_seqs.to(conf.device),
        'ext_idx_sql_seqs': ext_idx_sql_seqs.to(conf.device),
        'sql_seqs_lens': sql_seqs_lens,
        'batch_g_ids': batch_g_ids,
        'batch_g_nodes': batch_g_nodes,
        'batch_features': batch_features.to(conf.device),
        'batch_features_lens': batch_features_lens,
        'batch_fw_adj': batch_fw_adj.to(conf.device),
        'batch_bw_adj': batch_bw_adj.to(conf.device),
        'oov_list': oov_list
    }