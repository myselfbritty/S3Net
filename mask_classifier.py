#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as LY
import tensorflow.keras.backend as K
tf.keras.backend.set_image_data_format('channels_first')






class ArcFace(LY.Layer):
    
    def __init__(self, n_classes=783, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs, mode = 'margin'):
        if mode == 'margin':
            x, y = inputs
            # normalize feature
            x = tf.nn.l2_normalize(x, axis=-1)
            # normalize weights
            W = tf.nn.l2_normalize(self.W, axis=0)
            # dot product
            logits = x @ W
            # add margin
            # clip logits to prevent zero division when backward
            theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            target_logits = tf.cos(theta + self.m)
            
            logits = logits * (1 - y) + target_logits * y
            # feature re-scale
            logits *= self.s
            out = tf.nn.softmax(logits)
        else:
            x, _ = inputs
            
            # x = tf.nn.l2_normalize(x, axis=-1)
            # W = tf.nn.l2_normalize(self.W, axis=0)
            logits = x@self.W
            out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)




class MultiplyMask(LY.Layer):
    def __init__(self):
        super(MultiplyMask, self).__init__()
        self.ga = LY.GlobalAveragePooling2D()
        
    def call(self,feature, mask):
        op_features = []
        
        # for i in range(25):
        for i in range(25):
            current_mask = mask[:,i:i+1,:,:]
            current_feature = tf.multiply(feature, current_mask)
            current_feature = self.ga(current_feature)
            op_features.append(current_feature)
        op = tf.stack(op_features, axis=1)
        
        return op


class make_classifier(tf.keras.Model):
    
    def __init__(self, num_classes = 10, WEIGHT_DECAY = 1e-4):
        super(make_classifier, self).__init__()
        self.num_classes = num_classes
        
        self.resnet_model = keras.models.load_model('./pre-trained-weights/Stage_3/resnet_2.h5')
        self.conv1 = LY.Conv2D(256, 1)
        self.conv2 = LY.Conv2D(256, 1)
        self.conv3 = LY.Conv2D(256, 1)
        self.conv4 = LY.Conv2D(256, 1)
        
        self.up1 = LY.UpSampling2D(size=(8, 8), name="pre_cat_2")
        self.up2 = LY.UpSampling2D(size=(4, 4), name="pre_cat_3")
        self.up3 = LY.UpSampling2D(size=(2, 2), name="pre_cat_4")
        
        self.concat = LY.Concatenate(axis = 1)
        self.masking_layer  = MultiplyMask()
        
        self.dropout1 = LY.Dropout(0.35)

        self.dense1 = LY.Dense(1024, activation = 'relu')
        
        
        self.arc_layer = ArcFace(n_classes=self.num_classes, s=30, m=0.4)
        
    def call(self, inputs, mode = 'margin', training = True):
        if mode == 'margin':
            ip = inputs[0]
            mask_ip = inputs[1]
            y = inputs[2]
        else:
            ip = inputs[0]
            mask_ip = inputs[1]
            y = tf.zeros(shape = (ip.shape[0], 25, self.num_classes))
        op = self.resnet_model(ip)
        stage_2 = op[0]
        stage_3 = op[1]
        stage_4 = op[2]
        stage_5 = op[3]
        
        stage_5 = self.conv1(stage_5)
        stage_4 = self.conv2(stage_4)
        stage_3 = self.conv3(stage_3)
        stage_2 = self.conv4(stage_2)
        
        fp_2 = self.up1(stage_5)
        fp_3 = self.up2(stage_4)
        fp_4 = self.up3(stage_3)
        fp_5 = stage_2
        
        fp_o = self.concat([fp_2, fp_3, fp_4, fp_5])
        fp_o = self.masking_layer(fp_o, mask_ip)
        fp_o = self.dropout1(fp_o, training = training)
        x = self.dense1(fp_o)
        
        op = self.arc_layer([x,y], mode = mode)
        return op


    
