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
        keep_prob):

        self.input_x = x
        self.input_y = y

        inputs = tf.nn.embedding_lookup(embedding, self.input_x) 
        outputs = tf.reduce_mean(inputs,axis = 1)


        with tf.variable_scope("dropout"):
                outputs = tf.nn.dropout(outputs, keep_prob)
        
        with tf.variable_scope('outputs') :
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



