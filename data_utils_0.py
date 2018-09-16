#  分词

import  pickle,random
import numpy as np 

from config_0 import cfg

f = open('./seg_data.pkl','rb')

Data = pickle.load(f)
f.close()
articles,label = Data[0],Data[1]
label = [int(i)-1 for i in label]

max_len = cfg.max_len
#################################################
#  训练数据
train_size = cfg.train_size
train_articles = articles[:train_size]
train_label = label[:train_size]

test_size = len(label) - train_size 
test_articles = articles[-test_size:]
test_label = label[-test_size:]

######################################################

def batch_train_flow(batch_size):

    n_batches = train_size // batch_size

    x = train_articles[:n_batches * batch_size]
    y = train_label[:n_batches * batch_size]

    for j in range(n_batches):
        output_x = x[j*batch_size :(j+1)*batch_size]
        output_y = y[j*batch_size :(j+1)*batch_size]

        mmax = max_len

        input_x = np.zeros((batch_size,mmax),dtype= np.int)
        for k,xx in enumerate(output_x):
            input_x[k,-len(xx):] = xx[-mmax:]

        if cfg.model_type == 'han':           
            input_x = np.reshape(input_x, [batch_size, cfg.sentence_size, -1])

        yield input_x, output_y
######################################################################
# 测试集
def batch_test_flow(batch_size):
    
    n_batches = test_size // batch_size
    
    x = test_articles[:n_batches * batch_size]
    y = test_label[:n_batches * batch_size]

    for j in range(n_batches):
        output_x = x[j*batch_size :(j+1)*batch_size]
        output_y = y[j*batch_size :(j+1)*batch_size]

        mmax = max_len
                       
        input_x = np.zeros((batch_size,mmax),dtype= np.int)
        for k,xx in enumerate(output_x):
            input_x[k,-len(xx):] = xx[-mmax:]

        if cfg.model_type == 'han':           
            input_x = np.reshape(input_x, [batch_size, cfg.sentence_size, -1])

        yield input_x, output_y