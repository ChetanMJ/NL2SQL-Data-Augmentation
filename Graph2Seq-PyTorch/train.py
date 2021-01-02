#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from nltk.translate import bleu_score
import time

import config as conf
from preprocess import *
from sqlgraph_data import *
from aggregators import *
from model import *

def save_model(model, optimizer, fname="best_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, fname)
    
def load_saved_model(model_path, model, optimizer):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load optimizer state to GPU. Reference: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336031198
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
            
def idxs_to_sent(idxs, idx2word, oov_list=None):
    if oov_list is None:
        return ' '.join(list(map(lambda x : idx2word[str(x)], idxs)))
    else:
        tokens = []
        for i in idxs:
            if i < len(idx2word):
                tokens.append(idx2word[str(i)])
            else:
                i_oov = i - len(idx2word)
                tokens.append(oov_list[i_oov])
        return ' '.join(tokens)

def train(model, train_loader, dev_loader, num_epochs, criterion, optimizer, idx2word, scheduler=None, print_every=20, eval_every=200):
    batch_num = len(train_loader)
    model.train()
    
    running_avg_loss = 0
    best_bleu = 0.0
    for epochs in range(num_epochs):
        loss_sum = 0
        start_time = time.time()
        print("Epoch", epochs)
        for (batch_idx, collate_output) in enumerate(train_loader):
            model.train()
#             with torch.autograd.set_detect_anomaly(True):
            batch_g_nodes = collate_output['batch_g_nodes']
            batch_features = collate_output['batch_features']
            batch_fw_adj = collate_output['batch_fw_adj']
            batch_bw_adj = collate_output['batch_bw_adj']
            idx_seqs = collate_output['idx_seqs']
            ext_idx_seqs = collate_output['ext_idx_seqs']
            seqs_lens = collate_output['seqs_lens']
            idx_sql_seqs = collate_output['idx_sql_seqs']
            sql_seqs_lens = collate_output['sql_seqs_lens']
            ext_idx_sql_seqs = collate_output['ext_idx_sql_seqs']
            oov_list = collate_output['oov_list']

            output = model(batch_g_nodes, batch_features, batch_fw_adj, batch_bw_adj, \
                           idx_sql_seqs, ext_idx_sql_seqs, sql_seqs_lens, target_seq=idx_seqs, train=True)

            batch_size, nsteps, _ = output.size()

            preds = output.contiguous().view(batch_size * nsteps, -1)
            targets = ext_idx_seqs.contiguous().view(-1)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
            optimizer.step()

            running_avg_loss += loss.detach().item()
            if (batch_idx > 0 and batch_idx % print_every == 0):
                msg =  "{}/{} - loss : {:.4f}" \
                        .format(batch_idx, batch_num, running_avg_loss/print_every)
                print(msg)
                running_avg_loss = 0
                
            if batch_idx % eval_every == eval_every-1:
                with torch.no_grad():
                    eval_avg_loss, bleu = evaluate(model, dev_loader, criterion, idx2word, verbose=False)
                    print('Eval Avg Loss: %f' % eval_avg_loss)
                    if (bleu > best_bleu):
                        best_bleu = bleu
                        save_model(model, optimizer, "best_model.pth")

        end_time = time.time()
        print('Epoch Time:', end_time - start_time, 's')
        if scheduler is not None:
            scheduler.step()

def evaluate(model, dev_loader, criterion, idx2word, verbose=False):
    print("Start Eval")
    model.to(conf.device)
    model.eval()
    
    loss_sum = 0
    start_time = time.time()
    pds = []
    gts = []
    for (batch_idx, collate_output) in enumerate(dev_loader):
        batch_g_nodes = collate_output['batch_g_nodes']
        batch_features = collate_output['batch_features']
        batch_fw_adj = collate_output['batch_fw_adj']
        batch_bw_adj = collate_output['batch_bw_adj']
        idx_seqs = collate_output['idx_seqs']
        ext_idx_seqs = collate_output['ext_idx_seqs']
        seqs_lens = collate_output['seqs_lens']
        idx_sql_seqs = collate_output['idx_sql_seqs']
        sql_seqs_lens = collate_output['sql_seqs_lens']
        ext_idx_sql_seqs = collate_output['ext_idx_sql_seqs']
        oov_list = collate_output['oov_list']

        output = model(batch_g_nodes, batch_features, batch_fw_adj, batch_bw_adj, \
               idx_sql_seqs, ext_idx_sql_seqs, sql_seqs_lens, target_seq=idx_seqs, train=False)
        #print(output.size())
        #print(idx_seqs.size())
        batch_size, nsteps = idx_seqs.size()

        preds = output[:,:nsteps,:].contiguous().view(batch_size * nsteps, -1)
        targets = ext_idx_seqs.contiguous().view(-1)
        #print(preds.size())
        #print(targets.size())
        #print(preds)
        #print(targets)
        loss = criterion(preds, targets)
        
        loss = loss.item()
        loss_sum += loss
        if verbose:
            print('Batch %d | Validation Loss %f' % (batch_idx, loss))
        
        predicted_indices = output.cpu().numpy().argmax(axis=-1)
        for i in range(predicted_indices.shape[0]):
            idx_seq = predicted_indices[i]
            pd = idxs_to_sent(idx_seq, idx2word, oov_list[i]).split('<eos>')[0]
            #pd = pd.split('<sos>')[1]
            gt = idxs_to_sent(idx_seqs[i].cpu().numpy(), idx2word, oov_list[i]).split('<eos>')[0] 
            pds.append(pd.split(' '))
            gts.append([gt.split(' ')])
            if verbose:
                print("PD:", pd)
                print("GT:", gt)

    bleu = bleu_score.corpus_bleu(gts, pds)
    avg_loss = loss_sum / len(dev_loader)
    end_time = time.time()
    print('Eval Time:', end_time - start_time, 's')
    print('Eval Avg Loss:', avg_loss)
    print('Dev BLEU-4:', bleu * 100)
    model.train()
    return avg_loss, bleu

if __name__ == "__main__":
    print("device:", conf.device)
    # ## Preprocess

    print("Loading word index, word embeddings")
    if not os.path.isfile(conf.word2idx_path):
        print(conf.word2idx_path, "not found. Please first run python preprocess.py")

    with open(conf.word2idx_path, 'r') as f:
        word2idx = json.load(f)
    with open(conf.idx2word_path, 'r') as f:
        idx2word = json.load(f)
    with open(conf.embed_mat_path, 'r') as f:
        embed_mat = np.array(json.load(f))

    print("Loading data")
    train_set = SQLGraph_Dataset(conf.train_path, word2idx)
    dev_set = SQLGraph_Dataset(conf.dev_path, word2idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, collate_fn=collate, drop_last=True)

    model = Graph2Seq("train", conf, embed_mat).to(conf.device)

    num_epochs = 200
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 130, 160], gamma=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # load_saved_model('best_model.pth', model, optimizer)

    train(model, train_loader, dev_loader, num_epochs, criterion, optimizer, idx2word, scheduler=scheduler, print_every=50, eval_every=200)

    with torch.no_grad():
        evaluate(model, dev_loader, criterion, idx2word, verbose=False)
    
    save_model(model, optimizer, "final_model.pth")


