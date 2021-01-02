import json
from collections import OrderedDict, defaultdict
import spacy
import os
import numpy as np
import config as conf

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text.lower() for token in doc]

def get_vocab(embedding_dict, max_size=conf.max_vocab_size):
    special_words = {conf.PAD: conf.PAD_IDX, conf.SOS_TOKEN: conf.SOS_IDX, conf.EOS_TOKEN: conf.EOS_IDX, conf.OOV:conf.OOV_IDX}
#     idx2word = {v: k for k, v in word2idx.items()}
    wordcnt = defaultdict(int)
    def process_file(input_path, embedding_dict):
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                jo = json.loads(line, object_pairs_hook=OrderedDict)
                if 'text_tokens' in jo: # simple workaround for Spider dataset
                    seq = jo['text_tokens']
                else:
                    seq = word_tokenize(jo['text'])
                for w in seq:
                    if w in embedding_dict:
                        wordcnt[w] += 1

                for id in jo['g_ids_features']:
                    features = jo['g_ids_features'][id]
                    for w in word_tokenize(features):
                        if w in embedding_dict:
                            wordcnt[w] += 1


                sql_seq = jo['sql']
                for w in sql_seq:
                    if w in embedding_dict:
                        wordcnt[w] += 1

    process_file(conf.train_path, embedding_dict)
    process_file(conf.dev_path, embedding_dict)
    print(len(wordcnt))
    top_words = [(k, v) for k, v in sorted(wordcnt.items(), key=lambda item: item[1], reverse=True)][:max_size]
    word2idx = {k[0]: v+4 for v, k in enumerate(top_words)}
    word2idx.update(special_words)
    idx2word = {v : k for k, v in word2idx.items()}
    return word2idx, idx2word

def get_embedding(word2idx, embedding_dict, embedding_size=300):

    
    weights_matrix = np.zeros((len(word2idx), embedding_size))
    
    # oov_list = []
    
    for i, w in enumerate(word2idx.keys()):
        if w in embedding_dict:
            weights_matrix[i] = (embedding_dict[w])
        # else:
        #     # weights_matrix[i] = (np.random.normal(scale=0.01, size=300))
        #     oov_list.append(w)
            
    return weights_matrix
            
def save(fname, obj):
    print("Saving to", fname)
    with open(fname, 'w') as f:
        json.dump(obj, f)
        
if __name__ == "__main__":    
    embedding_dict = {}
    
    with open(conf.embedding_path, 'r', encoding="utf-8") as f:
        for l in f:
            lst = l.strip().split(' ')
            embedding_dict[lst[0]] = np.array(list(map(float, lst[1:])))

    word2idx, idx2word = get_vocab(embedding_dict)
    embed_mat = get_embedding(word2idx, embedding_dict, conf.embedding_size)
    print("vocab size:", len(word2idx))

    # if not os.path.isfile(word2idx_path):
    save(conf.word2idx_path, word2idx)
    save(conf.idx2word_path, idx2word)
    save(conf.embed_mat_path, embed_mat.tolist())
