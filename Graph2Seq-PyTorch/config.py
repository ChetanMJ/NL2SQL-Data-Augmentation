device = 'cuda'
train_path = 'data/spider_train.json'
dev_path = 'data/spider_dev.json'
test_path = None #'data/test_word_sql.data'
max_vocab_size = 10000
embedding_path = 'glove.6B.300d.txt'
embedding_size = 300
word2idx_path = 'word2idx.json'
idx2word_path = 'idx2word.json'
embed_mat_path = 'word_mat.json'

PAD = '<pad>'
OOV = '<oov>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
OOV_IDX = 3

word_embedding_dim = 100

train_batch_size = 32
dev_batch_size = 500
test_batch_size = 500

max_grad_norm = 5.0
l2_lambda = 0.000001
learning_rate = 0.001
epochs = 100
encoder_hidden_dim = 128
num_layers_decode = 1
word_size_max = 1
teacher_forcing_prob = 0.8
encoder_embedding_dropout = 0.5
decoder_embed_dropout = 0.5
decoder_lstm_dropout = 0.0 # no effect for single layer
bn = False

agg_dropout = 0.0

dropout = 0.0

path_embed_method = "lstm" # cnn or lstm or bi-lstm

deal_unknown_words = True

seq_max_len = 11
decode_max_len = 250

beam_max_len = 50

decoder_type = "greedy" # greedy, beam
beam_width = 10
attention = True
num_layers = 1 # 1 or 2

# the following are for the graph encoding method
weight_decay = 0.0000
sample_size_per_layer = 10
sample_layer_size = 1
max_unique_sample_layers = 7
hidden_layer_dim = 100
feature_max_len = 1
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"
concat = True

encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq"

lstm_in_gcn = "none" # before, after, none
