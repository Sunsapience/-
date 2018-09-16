# 分词
from easydict import EasyDict as edict

__C = edict()
cfg = __C 

#######################################################
# 文本参数
__C.vocab_size = 10000
__C.classes = 19
__C.max_len = 2000

###################################################
# 模型参数
__C.learning_rate = 0.0001
__C.embed_size = 200

__C.filter_sizes = [3,4,5]
__C.num_filters = 32

__C.hidden_size = 256
__C.num_layers = 2
__C.is_bidirection = True

__C.cnn_keep_prob = 0.5

__C.attention_size = 300

__C.rnn_keep_prob = 0.8
##################################################
# 训练参数
__C.batch_size = 200
__C.num_epoch = 15
__C.train_size = 90000

__C.sentence_size = 20

#########################################3

models_name = ['text_fast','text_cnn','text_rcnn','text_rnn','han']

__C.model_type= models_name[2]




