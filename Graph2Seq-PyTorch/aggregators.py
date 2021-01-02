import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# For output size to match input size, aggregator input size must be 2x output_size, if concat=True

# MLP, then max pooling to generate a single-vector neighbor encoding
class MaxPoolingAggregator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 dropout=0.0, bias=True, concat=True, mode='train'):
        super(MaxPoolingAggregator, self).__init__()
        self.mode = mode
        self.concat = concat
        self.dropout_i = nn.Dropout(dropout)
        self.dropout_n = nn.Dropout(dropout)
        self.mlp = nn.Linear(input_size, hidden_size, bias=bias) # could be multiple layer
        self.no = nn.Linear(hidden_size, output_size, bias=bias)
        self.io = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, input, neighbor_inputs):
        # input: [b, d]
        # neighbor_outputs: [b, max_degree, d]
        training = self.mode == 'train'
        input = self.dropout_i(input)
        neighbor_inputs = self.dropout_n(neighbor_inputs) 
        
        mask = (neighbor_inputs.sum(dim=-1) == 0).bool().unsqueeze(2) # [b, max_degree]
            
        neighbor_h = F.relu(self.mlp(neighbor_inputs)) # [b, max_degree, h]
        neighbor_h = neighbor_h.masked_fill(mask, value=0.0)
        neighbor_h, _ = torch.max(neighbor_h, dim=1)
        
        self_output = self.io(input)
        neighbor_output = self.no(neighbor_h)
    
        if not self.concat:
            output = self_output + neighbor_output
        else:
            output = torch.cat([self_output, neighbor_output], dim=-1) # [b, 2d]
        
        return F.relu(output)
    
# MLP, then max pooling to generate a single-vector neighbor encoding
# The version on their paper which is different from the one in theri code....
class MaxPoolingAggregatorPaper(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 dropout=0.0, bias=True, concat=True, mode='train'):
        super(MaxPoolingAggregatorPaper, self).__init__()
        self.mode = mode
        self.concat = concat
        self.dropout_i = nn.Dropout(dropout)
        self.dropout_n = nn.Dropout(dropout)
        self.mlp = nn.Linear(input_size, hidden_size, bias=bias) # could be multiple layer
        self.combine = nn.Linear(input_size + hidden_size, output_size * 2)
#         self.no = nn.Linear(hidden_size, output_size, bias=bias)
#         self.io = nn.Linear(input_size, output_size, bias=bias)
    
    def forward(self, input, neighbor_inputs):
        # input: [b, d]
        # neighbor_outputs: [b, max_degree, d]
        training = self.mode == 'train'
        input = self.dropout_i(input)
        neighbor_inputs = self.dropout_n(neighbor_inputs) 
        
        mask = (neighbor_inputs.sum(dim=-1) == 0).bool().unsqueeze(2) # [b, max_degree]
            
        neighbor_h = F.relu(self.mlp(neighbor_inputs)) # [b, max_degree, h]
        neighbor_h = neighbor_h.masked_fill(mask, value=0.0)
        neighbor_h, _ = torch.max(neighbor_h, dim=1)
        
#         self_output = self.io(input)
#         neighbor_output = self.no(neighbor_h)
    
#         if not self.concat:
#             output = self_output + neighbor_output
#         else:
#             output = torch.cat([self_output, neighbor_output], dim=-1) # [b, 2d]
        output = self.combine(torch.cat([input, neighbor_h], dim=-1))
        
        return F.relu(output)

# Simply take the average to generate a single-vector neighbor encoding
class MeanAggregator(nn.Module):
    def __init__(self, input_size, output_size, 
                 dropout=0.0, bias=True, concat=True, mode='train'):
        super(MeanAggregator, self).__init__()
        self.mode = mode
        self.concat = concat
        self.dropout = dropout
        self.no = nn.Linear(input_size, output_size, bias=bias)
        self.io = nn.Linear(input_size, output_size, bias=bias)
        
    def forward(self, input, neighbor_inputs, neighbor_lens=None):
        # input: [b, d]
        # neighbor_outputs: [b, max_degree, d]
        training = self.mode == 'train'
        input = F.dropout(input, p=self.dropout, training=training, inplace=True)
        neighbor_inputs = F.dropout(neighbor_inputs, p=self.dropout, training=training, inplace=True) 
        
        neighbor_means = torch.mean(neighbor_inputs, dim=1) # [b, d]
        # neighbor_means = torch.sum(neighbor_inputs, dim=1) / neightbor_lens.view(-1, 1) # if we want to mask padding
        
        self_output = self.io(input)
        neighbor_output = self.no(neighbor_means)
        
        if not self.concat:
            output = self_output + neighbor_output
        else:
            output = torch.cat([self_output, neighbor_output], dim=-1) # [b, 2d]
        
        return F.relu(output)
    
class GatedMeanAggregator(nn.Module):
    def __init__(self, input_size, output_size, 
                 dropout=0.0, bias=True, bn=True, concat=True, mode='train'):
        super(GatedMeanAggregator, self).__init__()
        self.mode = mode
        self.concat = concat
        self.dropout = dropout
        
        self.output_size = output_size * 2 if concat else output_size
        self.no = nn.Linear(input_size, output_size, bias=bias)
        self.io = nn.Linear(input_size, output_size, bias=bias)
        self.gate_l = nn.Linear(self.output_size, self.output_size)
        if bn:
            self.bn = nn.BatchNorm1d(self.output_size)
        
    def forward(self, input, neighbor_inputs, neighbor_lens=None):
        # input: [b, d]
        # neighbor_outputs: [b, max_degree, d]
        training = self.mode == 'train'
        input = F.dropout(input, p=self.dropout, training=training, inplace=True)
        neighbor_inputs = F.dropout(neighbor_inputs, p=self.dropout, training=training, inplace=True) 
                
        # the original implementation did not mask out "padding node" 
        # according to their comment this performs better?
        neighbor_means = torch.mean(neighbor_inputs, dim=1) # [b, d]
        # neighbor_means = torch.sum(neighbor_inputs, dim=1) / neightbor_lens.view(-1, 1) # if we want to mask padding
        
        self_output = self.io(input)
        neighbor_output = self.no(neighbor_means)
        
        if not self.concat:
            output = self_output + neighbor_output
        else:
            output = torch.cat([self_output, neighbor_output], dim=-1) # [b, 2d]
            
        gate = torch.cat([self_output, neighbor_output], dim=-1)
        gate = F.relu(self.gate_l(gate))
        
        if hasattr(self, 'bn'):
            output = self.bn(output)
        return gate * F.relu(output)
    
# Perform attention on neighbors to aggregate info?
class GatedAttnAggregator(nn.Module): # Is attention all you need?
    def __init__(self, input_size, output_size, 
                 dropout=0.0, bias=True, bn=True, concat=False, mode='train'):
        super(GatedAttnAggregator, self).__init__()
        self.mode = mode
        self.concat = concat
        self.dropout = dropout
        self.output_size = output_size * 2 if concat else output_size
        self.nq = nn.Linear(input_size, output_size)
        self.nk = nn.Linear(input_size, output_size)
        self.nv = nn.Linear(input_size, output_size)
        self.no = nn.Linear(output_size, output_size, bias=bias)
        self.io = nn.Linear(input_size, output_size, bias=bias)
        self.gate_l = nn.Linear(self.output_size, self.output_size)
        if bn:
            self.bn = nn.BatchNorm1d(self.output_size)
        
    # Simple bilinear attention. Could use scaled dot product attention?
    def forward(self, input, neighbor_inputs):
        training = self.mode == 'train'
        input = F.dropout(input, p=self.dropout, training=training, inplace=True)
        neighbor_inputs = F.dropout(neighbor_inputs, p=self.dropout, training=training, inplace=True) 
        # attention
        mask = (neighbor_inputs.sum(dim=-1) == 0).bool() # [b, max_degree]
        values = self.nv(neighbor_inputs)
        keys = self.nk(neighbor_inputs) # [b, max_degree, h]
        queries = self.nq(input).unsqueeze(2) # [b, h ,1]
        energies = torch.bmm(keys, queries).squeeze(2) # [b, max_degree]
        energies = energies.masked_fill(mask, value=1e-12)
        att_scores = F.softmax(energies, dim=-1)
        context = torch.bmm(att_scores.unsqueeze(1), values).squeeze(1) # [b, 1, max_degree] x [b, max_degree, h] -> [b, 1, h] -> [b, h]
        # update
        self_output = self.io(input)
        neighbor_output = self.no(context)
        
        if not self.concat:
            output = self_output + neighbor_output
        else:
            output = torch.cat([self_output, neighbor_output], dim=-1) # [b, 2d]
            
        gate = torch.cat([self_output, neighbor_output], dim=-1)
        gate = F.relu(self.gate_l(gate))
        
        if hasattr(self, 'bn'):
            output = self.bn(output)
        return gate * F.relu(output)
    