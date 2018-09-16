"""
Created 2018.9.12 20:33
@author: wlgzg
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn,cudnn_rnn

class Model():
    def __init__(self,
        x,
        y,
        is_training,
        embedding,
        batch_size,
        num_layers,
        hidden_size,
        attention_size,
        sentence_size,
        embed_size,
        keep_prob_1,
        keep_prob_2,
        classes,
        cell_type='lstm'):

        self.num_layers = num_layers
        self.drop = 1- keep_prob_1
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        # self.input_x = tf.placeholder(tf.int32, [batch_size, sentence_size, None])
        # self.input_y = tf.placeholder(tf.int64,[batch_size])

        self.input_x = x
        self.input_y = y

        inputs = tf.nn.embedding_lookup(embedding, self.input_x) 

        words_inputs = tf.reshape(inputs,[batch_size*sentence_size ,-1,embed_size])

        with tf.variable_scope('words_encode') as scope:
            word_outputs = self.rnn_encoder(words_inputs,scope)
            words_attn_outputs = self.attention(word_outputs,attention_size,scope)

            words_attn_outputs = tf.reshape(words_attn_outputs,[batch_size, sentence_size, -1])

            sentence_inputs= layers.dropout(words_attn_outputs,
                    keep_prob=keep_prob_2,
                    is_training=is_training)
        
        with tf.variable_scope('sentence_encode') as scope:
            sentence_outputs = self.rnn_encoder(sentence_inputs,scope)
            sentence_attn_outputs = self.attention(sentence_outputs,attention_size,scope)

            sentence_outputs= layers.dropout(sentence_attn_outputs,
                        keep_prob=keep_prob_2,
                        is_training=is_training)

        with tf.variable_scope('outputs') :
            logits = layers.fully_connected(sentence_outputs, 256, activation_fn = tf.tanh)
            logits = layers.fully_connected(logits, classes,activation_fn = None)
            self.prediction = tf.argmax(logits, axis=1)

        with tf.name_scope('losses'):            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=self.input_y)
            self.cost = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction,self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), 
                    name="accuracy")



    def rnn_encoder(self, inputs,scope=None):
        inputs = tf.transpose(inputs,perm=[1,0,2])
        with tf.variable_scope(scope or 'rnn_encoder'):

            DIRECTION = "bidirectional"
            if self.num_layers >= 2:
                num_layer = self.num_layers // 2
            else:
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


    def attention(self, inputs, size,scope=None):
        with tf.variable_scope(scope or 'attention'):
            attention_context = tf.get_variable(name='attention_context_vector',
                                    shape=[size],
                                    dtype=tf.float32)
            input_projection = layers.fully_connected(inputs, size,
                                    activation_fn=tf.tanh)

            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context), 
                                    axis=2, keepdims=True)
            attention_weights = tf.nn.softmax(vector_attn, axis=1)
            weighted_projection = tf.multiply(inputs, attention_weights)

            outputs = tf.reduce_sum(weighted_projection, axis=1)  

        return outputs 


    