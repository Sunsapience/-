# 单GPU

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from tensorflow.contrib import layers, cudnn_rnn
import tensorflow as tf

import numpy as np 

class Model():
    def __init__(self,
        x,
        y,
        embedding,
        is_bidirection,      
        classes,
        keep_prob_1,
        keep_prob_2,
        hidden_size,
        num_layers,
        cell_type='lstm'):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirection = is_bidirection
        self.cell_type = cell_type
        self.drop = 1- keep_prob_1
        self.input_x = x
        self.input_y = y

        inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        
        with tf.variable_scope('outputs') :
            outputs = self.rnn_encoder(inputs)
            outputs = tf.reduce_mean(outputs,axis = 1)
        

            with tf.variable_scope("dropout"):
                outputs = tf.nn.dropout(outputs, keep_prob_2)  

            logits = layers.fully_connected(outputs, 256,
                    activation_fn = tf.tanh)
            logits = layers.fully_connected(logits, classes,
                    activation_fn = None)
            self.prediction = tf.argmax(logits, axis=1)
        
        with tf.name_scope('losses'):            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=self.input_y)
            self.cost = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction,self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), 
                    name="accuracy")        
        

    def rnn_encoder(self, inputs):
        inputs = tf.transpose(inputs,perm=[1,0,2])
        with tf.variable_scope('rnn_encoder'):
            if self.is_bidirection:
                DIRECTION = "bidirectional"
                if self.num_layers >= 2:
                    num_layer = self.num_layers // 2
                else:
                    num_layer = self.num_layers
            else:
                DIRECTION = "unidirectional"
                num_layer = self.num_layers

            if self.cell_type.lower() == 'gru':
                cell = cudnn_rnn.CudnnGRU(num_layer,self.hidden_size,
                        direction=DIRECTION,dropout= 1-self.drop)
            else:
                cell = cudnn_rnn.CudnnLSTM(num_layer,self.hidden_size,
                        direction=DIRECTION,dropout= 1-self.drop)

            outputs,_ =  cell(inputs) 

        outputs = tf.transpose(outputs,perm=[1,0,2])
        return outputs

