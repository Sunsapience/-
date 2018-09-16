# 单 GPU
import logging,pickle
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import numpy as np 

import tensorflow as tf 

from data_utils_0 import batch_train_flow,batch_test_flow
from config_0 import cfg

from models.text_fast_0 import Model as text_fast
from models.text_cnn_0 import Model as text_cnn
from models.text_rcnn_0 import Model as text_rcnn  
from models.text_rnn_0 import Model as text_rnn
from models.text_sentence_0 import Model as han    

from mutil_gpu.fun import average_gradients
##########################################
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
    return _assign
##########################################
vocab_size = cfg.vocab_size + 1
classes = cfg.classes

max_len = cfg.max_len

learning_rate = cfg.learning_rate
embed_size = cfg.embed_size

filter_sizes = cfg.filter_sizes
num_filters = cfg.num_filters

hidden_size = cfg.hidden_size
num_layers = cfg.num_layers

cnn_keep_prob = cfg.cnn_keep_prob
rnn_keep_prob = cfg.rnn_keep_prob

batch_size = cfg.batch_size
num_epoch = cfg.num_epoch
train_size = cfg.train_size

sentence_size = cfg.sentence_size
is_bidirection = cfg.is_bidirection
attention_size = cfg.attention_size

#################################################
model_type=cfg.model_type
####################################################
class class_Model():
    def __init__(self,x,y,is_training):
       
        with tf.variable_scope("embedding"), tf.device('/cpu:0') :
            embedding = tf.get_variable("embedding", [vocab_size, embed_size])
        
        with tf.variable_scope('model'):
            if model_type == 'text_fast':
                self.model = text_fast(x,y,embedding,classes,cnn_keep_prob)
            elif model_type == 'text_cnn':
                self.model = text_cnn(x,y,
                        embedding,classes,embed_size,
                        max_len,filter_sizes, 
                        num_filters,cnn_keep_prob)
            elif model_type == 'text_rcnn':
                self.model = text_rcnn(x,y,
                        embedding,classes,embed_size,
                        cnn_keep_prob,rnn_keep_prob, 
                        hidden_size,num_layers)
            elif model_type == 'text_rnn':
                self.model = text_rnn(x,y,
                        embedding,is_bidirection,      
                        classes,rnn_keep_prob,cnn_keep_prob,hidden_size,
                        num_layers)
            else:
                self.model = han(x,y,is_training,
                    embedding,batch_size//2,
                    num_layers,hidden_size,
                    attention_size,sentence_size,
                    embed_size,
                    rnn_keep_prob,cnn_keep_prob,
                    classes)
        
        if not is_training:
            return
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.grads = self.optimizer.compute_gradients(self.model.cost)

# ###########################################################
def run_epoch(sess, input_x,input_y,model, data_queue, train_op, output_log):

    steps = 0
    feed_dict = {}
    accurate_1 = []
    for x,y in data_queue:
        feed_dict[input_x] = x
        feed_dict[input_y] = y
        
        accurate,loss, _ = sess.run([model.accuracy,model.cost,train_op], feed_dict=feed_dict)
        
        accurate_1.append(accurate)

        if output_log and steps % 10 == 0:
            logging.info('*****After {} steps,   loss_0 is {},   loss_1 is {},    accurate is {}'.format(
                steps,float('%.3f' % loss[0]),float('%.3f' % loss[1]),float('%.3f' % accurate)))        
        steps += 1
    return np.mean(accurate_1) 

# ##################################################
def main(_):
    with tf.device('/cpu:0'):
        initializer =tf.truncated_normal_initializer(stddev=0.1)
        tower_grads = []
        tower_loss = []
        train_accuracy = []
        test_accuracy = []
        reuse_vars = False

        if model_type == 'han':
            x = tf.placeholder(tf.int32, [batch_size, sentence_size, None])
            y = tf.placeholder(tf.int64,[None])
        else:
            x = tf.placeholder(tf.int32,[None, None])
            y = tf.placeholder(tf.int64,[None])

        for i in range(2):
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                _x = x[i * cfg.batch_size//2: (i+1) * cfg.batch_size//2]
                _y = y[i * cfg.batch_size//2: (i+1) * cfg.batch_size//2]

                with tf.variable_scope('LM', reuse=reuse_vars, initializer=initializer):
                    train_m = class_Model(_x,_y,True)

                with tf.variable_scope('LM', reuse=True, initializer=initializer):
                    eval_m = class_Model(_x,_y,False)

                reuse_vars = True
                tower_loss.append(train_m.model.cost)
                tower_grads.append(train_m.grads)

                train_accuracy.append(train_m.model.accuracy)
                test_accuracy.append(eval_m.model.accuracy)

        tower_grads = average_gradients(tower_grads)
        train_op = train_m.optimizer.apply_gradients(tower_grads)

        test_acc = tf.reduce_mean(test_accuracy)
        train_acc = tf.reduce_mean(train_accuracy)

        saver = tf.train.Saver(max_to_keep=15)
        with tf.Session( config = tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer()) 

            for i in range(num_epoch):
                train_data = batch_train_flow(cfg.batch_size)           
                test_data = batch_test_flow(cfg.batch_size)

                logging.info('In iteration: {}'.format(i+1))

                train_accurate = run_epoch(sess, x,y,train_m.model, 
                        train_data, train_op, True) 
                logging.info('##################################')
                test_accurate = run_epoch(sess, x,y,eval_m.model,
                        test_data,tf.no_op(), True) 
                
                logging.info('##################################')

                logging.info('Train accurate: {}'.format(float('%.3f' % train_accurate)))                        
                logging.info('Test accurate: {}'.format(float('%.3f' % test_accurate)))
                

                logging.info('**********************************')
                print('**********************************')
                
                print('In iteration: {}'.format(i+1))
                print('Train accurate: {}'.format(float('%.3f' % train_accurate)))                        
                print('Test accurate: {}'.format(float('%.3f' % test_accurate)))

                saver.save(sess,'./save/muil_model_sentence-',global_step=i+1)

if __name__ == '__main__':
    tf.app.run()
        


        
        





