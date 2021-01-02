import numpy as np
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
import config as conf
from aggregators import *
# from decoder import *
import sys
if sys.version > '3':
	from queue import PriorityQueue
	from queue import Queue
else:
    from Queue import PriorityQueue
    from Queue import Queue
import random
from torch_scatter import scatter_max

INF = 1e12

class BeamNode(object):
    def __init__(self, hidden,context, previous_node, decoder_input, attn, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.log_prob = log_prob
        self.length = length 
        self.prev_context = context
    def __lt__(self,other): 
        return self.log_prob/float(self.length-1+1e-6) < other.log_prob/float(other.length-1+1e-6)
    def eval(self):
        return self.log_prob/float(self.length-1+1e-6)

class Graph2Seq(nn.Module):
    # word_mat: numpy embedding matrix [vocab_size, embedding_size]
    def __init__(self, mode, conf, word_mat):
        super(Graph2Seq,self).__init__()
        self.conf = conf
        self.mode = mode
        # self.word_vocab_size = conf.word_vocab_size
        self.l2_lambda = conf.l2_lambda
        self.path_embed_method = conf.path_embed_method #lstm
        # self.word_embedding_dim = conf.word_embedding_dim
        self.word_embedding_dim = conf.hidden_layer_dim
        self.encoder_hidden_dim = conf.encoder_hidden_dim

        # the setting for the GCN
        self.num_layers_decode = conf.num_layers_decode
        self.num_layers = conf.num_layers
        self.graph_encode_direction = conf.graph_encode_direction
        self.sample_layer_size = conf.sample_layer_size
        self.hidden_layer_dim = conf.hidden_layer_dim
        self.concat = conf.concat

        # the setting for the decoder
        self.beam_width = conf.beam_width
        self.decoder_type = conf.decoder_type
        self.seq_max_len = conf.seq_max_len
        self.teacher_forcing_prob = conf.teacher_forcing_prob

        #self.sample_size_per_layer = tf.shape(self.fw_adj_info)[1]
        #self.single_graph_nodes_size = tf.shape(self.batch_nodes)[1]

        self.attention = conf.attention
        self.dropout = conf.dropout
        self.fw_aggregators = []
        self.bw_aggregators = []

        self.if_pred_on_dev = False

        self.vocab_size = len(word_mat)
        self.embed_layer = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0).to(conf.device)
        self.embed_layer.weight.data.copy_(torch.from_numpy(word_mat))
        self.embed_layer.weight.requires_grad = False
        self.encoder = GraphEncoder(conf, self.embed_layer, conf.path_embed_method)
        concat_dim_mul = 2 if conf.concat else 1
        dir_dim_mul = 2 if conf.graph_encode_direction == "bi" else 1
        self.decoder = GraphDecoder(conf, self.embed_layer, conf.encoder_hidden_dim * concat_dim_mul * dir_dim_mul, 
            len(word_mat), dropout_p=conf.decoder_embed_dropout, num_layers=1)

        self.SOS_IDX = conf.SOS_IDX
        self.EOS_IDX = conf.EOS_IDX
        self.topk = 1

    def forward(self, batch_nodes, node_features, fw_adj, bw_adj, idx_sql_seqs, ext_idx_sql_seqs, sql_seqs_lens, target_seq=None, train=True, random=False):
#         if train: assert(target_seq is not None)
        graph_hidden, graph_embedding, max_len, seqs_encoding, seqs_encoding_mask = self.encoder(batch_nodes, node_features, fw_adj, bw_adj, idx_sql_seqs, sql_seqs_lens) # graph_hidden = node embeddings
        batch_size, node_seq_max_len, _ = graph_hidden.size()
        decoder_hidden = (graph_embedding.unsqueeze(0), torch.zeros(graph_embedding.size()).unsqueeze(0).to(conf.device)) # h, c
        encoder_outputs = graph_hidden
        if train:
            out_max_len = target_seq.size(1)
            decoder_input = torch.LongTensor([self.SOS_IDX for _ in range(batch_size)]).to(conf.device)
            prev_context = torch.zeros((batch_size, graph_hidden.size(-1))).to(conf.device)
            predictions = []
            for idx in range(out_max_len):
                output, prev_context, decoder_hidden, attn_weights = self.decoder(decoder_input, prev_context, decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
                predictions.append(output.unsqueeze(1)) # [b, 1, vocab_size]
                if random.random() < self.teacher_forcing_prob:
                    decoder_input = target_seq[:, idx] # [b, ]
                else:
                    y_i = output.argmax(dim=-1).detach().cpu().numpy()
                    latest_tokens = [idx if idx < self.vocab_size else conf.OOV_IDX for idx in y_i]
                    decoder_input = torch.LongTensor(latest_tokens).to(conf.device)
            decoded_batch = torch.cat(predictions, dim=1)
        elif random:
            decoded_batch = self.random_decode(decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
        elif self.decoder_type == 'beam':
            #raise NotImplementedError # TODO
            decoded_batch = self.beam_decode(decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
        else: # greedy
            decoded_batch = self.greedy_decode(decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
        return decoded_batch # [b, t, vocab_size]
    
    def random_decode(self, decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs):
        batch_size = decoder_hidden[0].size(1)
        out_max_len = conf.decode_max_len
        decoder_input = torch.LongTensor([self.SOS_IDX for _ in range(batch_size)]).to(conf.device)
        prev_context = torch.zeros((batch_size, decoder_hidden[0].size(-1))).to(conf.device)
        predictions = []
        for idx in range(out_max_len):
            output, prev_context, decoder_hidden, attn_weights = self.decoder(decoder_input, prev_context, decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
            predictions.append(output.unsqueeze(1))
            # y_i = output.argmax(dim=-1).detach().cpu().numpy() # b, vocab_size
            y_i = torch.multinomial(F.softmax(output, dim=-1), 1).squeeze(-1).detach().cpu().numpy()
            latest_tokens = [idx if idx < self.vocab_size else conf.OOV_IDX for idx in y_i]
            decoder_input = torch.LongTensor(latest_tokens).to(conf.device)
        return torch.cat(predictions, dim=1) # [b, t, vocab_size]
    
    def greedy_decode(self, decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs):
        batch_size = decoder_hidden[0].size(1)
        out_max_len = conf.decode_max_len
        decoder_input = torch.LongTensor([self.SOS_IDX for _ in range(batch_size)]).to(conf.device)
        prev_context = torch.zeros((batch_size, decoder_hidden[0].size(-1))).to(conf.device)
        predictions = []
        for idx in range(out_max_len):
            output, prev_context, decoder_hidden, attn_weights = self.decoder(decoder_input, prev_context, decoder_hidden, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs)
            predictions.append(output.unsqueeze(1))
            y_i = output.argmax(dim=-1).detach().cpu().numpy()
            latest_tokens = [idx if idx < self.vocab_size else conf.OOV_IDX for idx in y_i]
            decoder_input = torch.LongTensor(latest_tokens).to(conf.device)
        return torch.cat(predictions, dim=1) # [b, t, vocab_size]

    # def greedy_decode(self,decoder_hidden,encoder_outputs,decoder):
    #     batch_size = decoder_hidden.size()[1]
    #     decoded_batch = torch.zeros((batch_size,self.seq_max_len)).to(conf.device)
    #     decoder_input = torch.LongTensor([[self.SOS_IDX] for _ in range(batch_size)])

    #     for idx in range(self.seq_max_len):
    #         decoder_output, decoder_hidden, attn_weight = decoder(decoder_input, decoder_hidden,encoder_outputs)
    #         topv, topi = decoder_output.data.topk(1)  # get candidates
    #         topi = topi.view(-1)
    #         decoded_batch[:, idx] = topi
    #         decoder_input = topi.detach().view(-1, 1)
    #     return decoded_batch

    def beam_decode(self,decoder_hiddens,encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs):
        batch_size = decoder_hiddens[0].size(1)
        decoded_batch = torch.zeros([batch_size, conf.decode_max_len+1, torch.max(ext_idx_sql_seqs)+1]).to(conf.device)
        sent_cou = 1 # number of sentence generate
        for idx in range(batch_size):
            seqs_enc = seqs_encoding[idx, :, :].unsqueeze(0)
            seqs_enc_mask = seqs_encoding_mask[idx, :].unsqueeze(0)
            ext_idx_sql = ext_idx_sql_seqs[idx, :].unsqueeze(0)
            decoder_hidden =  (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
            prev_context = torch.zeros((1, decoder_hidden[0].size(-1))).to(conf.device)
            encoder_output = encoder_outputs[idx,:, :].unsqueeze(0).to(conf.device)
            decoder_input = torch.LongTensor([self.SOS_IDX]).to(conf.device)
            root = BeamNode(decoder_hidden,prev_context,None,decoder_input,None,0,1)
            que = Queue()
            que.put((1,-root.eval(),root))
            end_nodes = []
            vocab_size = 0
            while_cnt = 0
            while not que.empty():
                candidates = []
                while_cnt += 1
                for _ in range(que.qsize()):
                    _, score, node = que.get()
                    decoder_inp = node.decoder_input if node.decoder_input.item() < self.vocab_size else torch.tensor([conf.OOV_IDX]).to(conf.device)
                    decoder_hid = node.hidden
                    prev_cont = node.prev_context
                    if decoder_inp.item() == self.EOS_IDX or node.length > self.conf.beam_max_len:
                        end_nodes.append((score,node))
                        continue
                    log_prob, prev_cont, hidden, attn = self.decoder(decoder_inp,prev_cont,decoder_hid,encoder_output, seqs_enc, seqs_enc_mask, ext_idx_sql)
                    vocab_size = log_prob.size(1) 
                    log_prob, indices = log_prob[0].topk(self.beam_width)
                    for k in range(self.beam_width):
                        index = indices[k].unsqueeze(0)
                        log_p = log_prob[k].item()
                        new_node = BeamNode(hidden,prev_cont,node,index,attn,node.log_prob+log_p,node.length+1)
                        score = node.log_prob+log_p
                        candidates.append((node.length+1,score, new_node))
                candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
                leng_can = min(len(candidates),self.beam_width)
                for k in range(leng_can):
                    leng, score, nn = candidates[k]
                    que.put((leng,score,nn))
            #back-track
            if len(end_nodes) == 0:
                _, score, end_node = que.get()
            else:
                score, end_node = end_nodes[0]
            utterance = []
            utterance.append(end_node.decoder_input)
            while end_node.previous_node != None:
                end_node = end_node.previous_node
                utterance.append(torch.LongTensor([end_node.decoder_input]).to(conf.device))
            utterance = utterance[::-1]
            utterance = utterance[1:]
            while len(utterance) < self.conf.decode_max_len+1:
                utterance.append(torch.LongTensor([self.EOS_IDX]).to(conf.device))
            utterance = torch.cat(utterance,dim=0).to(conf.device).unsqueeze(1)
            
            utterance = (torch.zeros(utterance.size(0),vocab_size)).to(conf.device).scatter_(1,utterance,1)
            decoded_batch[idx, :, :utterance.size(-1)] = utterance
        return decoded_batch
        
class GraphEncoder(nn.Module):
    def __init__(self, conf, embed_layer, path_embed_method):
        super(GraphEncoder, self).__init__()
        self.mode = "train"
        self.sample_size_per_layer = conf.sample_size_per_layer # i.e. max degree
        self.sample_layer_size = conf.sample_layer_size # i.e. num sample layers
        self.max_unique_sample_layers = conf.max_unique_sample_layers
        self.graph_encode_direction = conf.graph_encode_direction
        self.embedding_size = conf.embedding_size
        self.hidden_size = conf.encoder_hidden_dim
        
        self.embed_layer = embed_layer
        self.seq_embedding_dropout = nn.Dropout(p=conf.encoder_embedding_dropout)
        self.seq_encoder = nn.LSTM(self.embedding_size, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.node_feature_encoder = nn.LSTM(self.embedding_size, self.hidden_size, 1, bidirectional=False, batch_first=True) # bidir ?
        self.fw_aggregators = []
        self.bw_aggregators = []

        for l in range(self.sample_layer_size):
            dim_mul = 1 if l == 0 else 2 # because output is concatenated
            if l > self.max_unique_sample_layers:
                fw_aggregator = self.fw_aggregators[self.max_unique_sample_layers-1]
            else:
                fw_aggregator = GatedAttnAggregator(self.hidden_size * dim_mul, self.hidden_size, 
                                                    dropout=conf.agg_dropout, bias=True, bn=conf.bn, concat=True, mode='train') # GatedAttnAggregator, MaxPoolingAggregator
                self.fw_aggregators.append(fw_aggregator)
            
            if self.graph_encode_direction == "bi":
                if l > self.max_unique_sample_layers:
                    bw_aggregator = self.bw_aggregators[self.max_unique_sample_layers-1]
                else:
                    bw_aggregator = GatedAttnAggregator(self.hidden_size * dim_mul, self.hidden_size, # self.hidden_size,
                                                        dropout=conf.agg_dropout, bias=True, bn=conf.bn, concat=True, mode='train') # GatedAttnAggregator, MaxPoolingAggregator
                    self.bw_aggregators.append(bw_aggregator)

        self.fw_aggregators = nn.ModuleList(self.fw_aggregators)
        self.bw_aggregators = nn.ModuleList(self.bw_aggregators)

    def train_mode(self):
        self.mode = "train"
        for agg in self.fw_aggregators:
            agg.mode = "train"
        for agg in self.bw_aggregators:
            agg.mode = "train"
    
    def eval_mode(self):
        self.mode = "eval"
        for agg in self.fw_aggregators:
            agg.mode = "eval"
        for agg in self.bw_aggregators:
            agg.mode = "eval"
            
    def forward(self, batch_nodes, node_features, fw_adj, bw_adj, idx_sql_seqs, sql_seqs_lens):
        # import pdb; pdb.set_trace()
        seqs_embedded = self.embed_layer(idx_sql_seqs)
        seqs_embedded = self.seq_embedding_dropout(seqs_embedded)
        seqs_packed = pack_padded_sequence(seqs_embedded, sql_seqs_lens, batch_first=True, enforce_sorted=False)
        seqs_encoding, _ = self.seq_encoder(seqs_packed)
        seqs_encoding, _ = pad_packed_sequence(seqs_encoding, batch_first=True)
        seqs_encoding_mask = (idx_sql_seqs == 0).bool() # [b, t]

        batch_size, seq_len = node_features.size()
        output = self.embed_layer(node_features)
        node_output, _ = self.node_feature_encoder(output) # features are short, no need to pack
        node_embedding = node_output[:,-1,:] # take the last timestep as initial node embedding ?
        
        fw_hidden = node_embedding
        bw_hidden = node_embedding.clone()
        embedded_node_rep = torch.cat([node_embedding.clone(), torch.zeros([1, self.hidden_size]).to(conf.device)], dim=0) # add a row of zero for PAD node
        
        fw_sampled_neighbors = fw_adj[:-1,:self.sample_size_per_layer] # ignore PAD node
        bw_sampled_neighbors = bw_adj[:-1,:self.sample_size_per_layer]
        
#         fw_sampled_neighbors_len = fw_adj.size(0)
#         bw_sampled_neighbors_len = bw_adj.size(0)
        
        for l in range(self.sample_layer_size):
            dim_mul = 1 if l == 0 else 2 # because output is concatenated
            fw_aggregator = self.fw_aggregators[min(l, self.max_unique_sample_layers-1)]
                
            if l == 0:
                # the PAD node will get zero embeddings
                neighbor_hiddens = F.embedding(fw_sampled_neighbors, embedded_node_rep)
            else:
                neighbor_hiddens = F.embedding(fw_sampled_neighbors,
                    torch.cat([fw_hidden, torch.zeros([1, dim_mul * self.hidden_size]).to(conf.device)], dim=0))
                
            fw_hidden = fw_aggregator(fw_hidden, neighbor_hiddens)
            
            if self.graph_encode_direction == "bi":
                bw_aggregator = self.bw_aggregators[min(l, self.max_unique_sample_layers-1)]

                if l == 0:
                    neighbor_hiddens = F.embedding(bw_sampled_neighbors, embedded_node_rep)
                else:
                    neighbor_hiddens = F.embedding(bw_sampled_neighbors,
                        torch.cat([bw_hidden, torch.zeros([1, dim_mul * self.hidden_size]).to(conf.device)], dim=0)) # the PAD node will get zero embeddings

                bw_hidden = bw_aggregator(bw_hidden, neighbor_hiddens)
            
        # Graph Embedding: max pooling
        if self.graph_encode_direction == "bi":
            hidden = torch.cat([fw_hidden, bw_hidden], axis=-1)
        else:
            hidden = fw_hidden
        
        hidden = F.relu(hidden) # [b, out_h]
        
        out_hidden_size = hidden.size(1)
        num_graphs = len(batch_nodes)
        max_len = max([len(g) for g in batch_nodes])
        graph_hidden = torch.zeros([num_graphs, max_len, out_hidden_size]).to(conf.device)
        for i, g_node_idxs in enumerate(batch_nodes):
            graph_hidden[i,:len(g_node_idxs),:] = hidden[g_node_idxs[0]:g_node_idxs[-1]+1]
        
        graph_embedding, _ = torch.max(graph_hidden, dim=1) # [num_g, out_h]
        return graph_hidden, graph_embedding, max_len, seqs_encoding, seqs_encoding_mask

class GraphDecoder(nn.Module):
    def __init__(self, conf, embed_layer, hidden_size, output_size, dropout_p, num_layers):
        super(GraphDecoder, self).__init__()
        self.embedding_size = conf.embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.embed_layer = embed_layer
        self.reduce_layer = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)
        self.key_network = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.seq_key_network = nn.Linear(conf.encoder_hidden_dim * 2, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size) 

    def forward(self, input, prev_context, hidden_states, encoder_outputs, seqs_encoding, seqs_encoding_mask, ext_idx_sql_seqs):
        # input [b, ], hidden [b, h], encoder_outputs [b, max_seq_len, h]
        # prev_context [b, h], initially zero
        embedded = self.embed_layer(input) # [b, embedding_size]
        embedded = self.dropout(embedded)
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=1)).unsqueeze(1)
        output, hidden_states = self.lstm(lstm_inputs, hidden_states)
        output = output.squeeze(1)
        mask = (encoder_outputs.sum(dim=-1) == 0).bool()
        keys = self.key_network(encoder_outputs) # [b, max_seq_len, h]
        queries = output.unsqueeze(2) # [b, h, 1]
        #print(encoder_outputs.size())
        #print(keys.size())
        #print(queries.size())
        energies = torch.bmm(keys, queries).squeeze(2) # [b, max_seq_len]
        energies = energies.masked_fill(mask, value=1e-12)
        attn_weights = F.softmax(energies, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1) # [b, 1, max_seq_len] x [b, max_seq_len, h] -> [b, 1, h] -> [b, h]
        output = self.attn_combine(torch.cat([output, context], dim=1)) # [b, 2h] -> [b, h]
        # output = F.relu(output)
        output = torch.tanh(output) # not sure which one is better though
        output = self.out(output)

        # calculate energies (raw attention scores) over input sql sequence
        seq_keys = torch.tanh(self.seq_key_network(seqs_encoding)) # [b, max_sql_seq_len, h]
        seq_energies = torch.bmm(seq_keys, queries).squeeze(2) # [b, max_sql_eq_len]
        seq_energies = seq_energies.masked_fill(seqs_encoding_mask, value=1e-12)
        # combine copy and generation
        vocab_size = self.embed_layer.weight.size(0)
        batch_size = input.size(0)
        num_oov = max(torch.max(ext_idx_sql_seqs - vocab_size + 1), 0)
        zeros = torch.zeros((batch_size, num_oov), device=conf.device)
        extended_output = torch.cat([output, zeros], dim=1)
        copy_scores = torch.zeros_like(extended_output) - INF
        copy_scores, _ = scatter_max(seq_energies, ext_idx_sql_seqs, out=copy_scores) # only use the max score (use scatter_add to combine scores)
        copy_scores = copy_scores.masked_fill(copy_scores == -INF, 0)
        output = extended_output + copy_scores
        output = output.masked_fill(output == 0, -INF)
        
        output = F.log_softmax(output, dim=1)
        return output, context, hidden_states, attn_weights