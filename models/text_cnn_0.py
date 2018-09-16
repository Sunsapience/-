# 单GPU

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from tensorflow.contrib import layers, rnn
import tensorflow as tf

import numpy as np 

class Model():
    def __init__(self, 
        x,
        y,
        embedding,
        classes,
        embed_size,
        sequence_length,
        filter_sizes, 
        num_filters,
        keep_prob):

        self.input_x = x
        self.input_y = y
 
        inputs = tf.nn.embedding_lookup(embedding, self.input_x) #[batch_size,max_len,embed_size]
        inputs_expanded = tf.expand_dims(inputs, -1) #[batch_size,max_len,embed_size,1]
        
        pooled_outputs = []
        for filter_size in filter_sizes: # [3,4,5]
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embed_size, 1, num_filters]
                W = tf.get_variable('W',filter_shape) 
                b = tf.get_variable('b',[num_filters])

                conv = tf.nn.conv2d(inputs_expanded,W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, 
                        ksize=[1, sequence_length - filter_size +1, 1, 1], 
                        strides=[1, 1, 1, 1], 
                        padding='VALID', 
                        name="pool")
                
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3) # [batch_size,1,1,32*3]

        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  #[batch_size,32*3]   


        with tf.variable_scope("dropout"):
                h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

        with tf.variable_scope('outputs') :
            logits = layers.fully_connected(h_pool_flat, 256,
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


