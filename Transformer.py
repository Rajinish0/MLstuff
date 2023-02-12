'''
mostly from tensorflow's tutorial (before they made it more abstract)
'''

import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from tensorflow import keras
from keras import backend as K



examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']



for pt, en in train_examples.take(1):
  print("Portuguese: ", pt.numpy().decode('utf-8'))
  print("English:   ", en.numpy().decode('utf-8'))



train_en = train_examples.map(lambda pt, en: en)
train_pt = train_examples.map(lambda pt, en: pt)


engVocab = dict()
for i, item in enumerate(train_en):
#     print("{:.2f}".format(i/L), end='\r')
    for word in item.numpy().split():
        engVocab.setdefault(word.decode(), 0)
        engVocab[word.decode()] += 1

pt_vocab = dict()
for i, item in enumerate(train_pt):
    #print("{:.2f}".format(i/L), end='\r')
    for word in item.numpy().split():
        pt_vocab.setdefault(word.decode(), 0)
        pt_vocab[word.decode()] += 1
#         pt_vocab.add(word.decode())


engVocab = list(map(lambda x : x[0], sorted(list(engVocab.items()), key=lambda x : x[1])[::-1][:8000]))
pt_vocab = list(map(lambda x : x[0], sorted(list(pt_vocab.items()), key=lambda x : x[1])[::-1][:8000]))


engVocab = list(engVocab)
pt_vocab = list(pt_vocab)

print(engVocab[:5], pt_vocab[:5])
print(len(engVocab), len(pt_vocab))



engVocab = ['<pad>'] + ["[START]", "[END]"] + engVocab
pt_vocab = ['<pad>'] + ["[START]", "[END]"] + pt_vocab


indicies = tf.range(len(engVocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(engVocab, indicies)
num_oov_buckets = 500
tableEng = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)



indicies2 = tf.range(len(pt_vocab), dtype=tf.int64)
table_init1 = tf.lookup.KeyValueTensorInitializer(pt_vocab, indicies2)
num_oov_buckets = 500
tablePt = tf.lookup.StaticVocabularyTable(table_init1, num_oov_buckets)


def preprocess(x, y):
    x = tf.strings.split("[START] " + x + " [END]")
    y = tf.strings.split("[START] " + y)
    return (x.to_tensor(default_value = '<pad>'), y.to_tensor(default_value='<pad>'))



def process(x, y):
    return (tablePt.lookup(x), tableEng.lookup(y))



def finProcess(x, y):
    return ((x, y[:, :-1]), y[:, 1:])


batch_size = 8
data = train_examples.batch(batch_size).map(preprocess).map(process).map(finProcess).prefetch(1)
val_data = val_examples.batch(batch_size).map(preprocess).map(process).map(finProcess).prefetch(1)


@tf.function
def calcAttention(q,k,v, mask = None):
    ## assumes k is already transposed.
    size = tf.cast(tf.shape(q)[-1],tf.float32)
    similarities = tf.matmul(q, k)/tf.sqrt(size)
    if mask is not None:
        similarities += mask*(-10**9)
    acts = tf.keras.activations.softmax(similarities,axis=-1)
    return tf.matmul(acts, v)


@tf.function
def posEncoding(seqLen, dimModel):
    pos = np.arange(seqLen)[:, np.newaxis]
    dis = np.arange(dimModel)[np.newaxis, :]
    nums = pos/(10000**(2*dis/dimModel))
    nums[:, ::2] = np.sin(nums[:, ::2])
    nums[:, 1::2] = np.cos(nums[:, 1::2])
    return tf.cast(nums[np.newaxis, ...], tf.float32)



class MultiHeadedAttention(keras.layers.Layer):
    def __init__(self, qkDim, numHeads=4, **kwargs):
        super().__init__(**kwargs)
        self.qkDim = qkDim
        self.numHeads=numHeads
        
        self.Wq = keras.layers.Dense(numHeads*qkDim)
        self.Wk = keras.layers.Dense(numHeads*qkDim)
        self.Wv = keras.layers.Dense(numHeads*qkDim)
        
        self.W0 = keras.layers.Dense(1)
#         self.dropout = keras.layers.Dropout(0.2)
    
    def splitHead(self, vec):
        batch_size = tf.shape(vec)[0]
        return tf.reshape(vec, [batch_size, self.numHeads, -1, self.qkDim])
    
    def call(self, q, k, v, mask=None):
        q, k, v = (self.splitHead(self.Wq(q)), 
                   self.splitHead(self.Wk(k)),
                   self.splitHead(self.Wv(v)))
        
        return self.W0(tf.transpose(calcAttention(q, tf.transpose(k, [0, 1, 3, 2]), v, mask), [0, 2, 3, 1]))[..., -1]
        

class EncoderLayer(keras.layers.Layer):
    def __init__(self, qkDim=64, numHeads=4, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadedAttention(qkDim, numHeads)
        self.layerNorm = tf.keras.layers.Normalization()
        
        self.nn=  keras.models.Sequential([
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(qkDim),
            keras.layers.Dropout(0.2),
        ])
        self.layerNorm2 = tf.keras.layers.Normalization()
        
    def call(self, x, mask=None):
        out = self.mha(x, x, x, mask)
        out1= self.layerNorm(out + x)
        
        out = self.nn(out1)
        out = self.layerNorm2(out + out1)
        return out


class Encoder(keras.layers.Layer):
    def __init__(self, vocabLen,embdDim=256, numEncLayers= 6, numHeads=4, **kwargs):
        super().__init__(**kwargs)
        self.embdDim = embdDim
        self.embd = keras.layers.Embedding(vocabLen+num_oov_buckets, embdDim, mask_zero=True)
        self.encs = [EncoderLayer(embdDim) for i in range(numEncLayers)]
    
    def call(self, X, mask=None):
        seqLen = tf.shape(X)[1]
        out = self.embd(X)
        ## TO DO
        # replace 2048 with maxSeqLen
        out += posEncoding(2048, self.embdDim)[:, :seqLen, :]
        for each in self.encs:
            out = each(out, mask)
        return out
        
        
def createPadMask(vec):
    return tf.cast(K.equal(vec, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

def createMask(size):
    return (1-tf.linalg.band_part(tf.ones([size, size]), -1, 0) )[tf.newaxis, tf.newaxis, ...]



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, qkDim=64, numHeads=4, **kwargs):
        super().__init__(**kwargs)
        self.mh1 = MultiHeadedAttention(qkDim, numHeads)
        self.layerNorm1 = keras.layers.LayerNormalization()
        
        self.mh2 = MultiHeadedAttention(qkDim, numHeads)
        self.layerNorm2 = keras.layers.LayerNormalization()
        
        self.nn = keras.models.Sequential([
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(qkDim),
            keras.layers.Dropout(0.2)
        ])
        self.layerNorm3 = keras.layers.LayerNormalization()
        
        
    def call(self, X, encOutputs, mask, mask2):
        ## self attention
        out = self.mh1(X, X, X, mask)
        out1 = self.layerNorm1(out + X)
        
        ##cross Attention (don't need mask)
        ## actually need the padding mask here
        out = self.mh2(out1, encOutputs, encOutputs, mask2)
        out1 = self.layerNorm2(out + out1)
        
        ##dense
        out = self.nn(out1)
        out = self.layerNorm3(out + out1)
        
        return out
        
        

class Decoder(keras.layers.Layer):
    def __init__(self, vocabLen, outDim, embdDim=256, numDecLayers=6, numHeads=4, **kwargs):
        super().__init__(**kwargs)
        self.embd = keras.layers.Embedding(vocabLen+num_oov_buckets, embdDim)
        self.posEnc = posEncoding(2048, embdDim)
        self.decLayers = [DecoderLayer(embdDim, numHeads) for i in range(numDecLayers)]
        self.dense = keras.layers.Dense(outDim)
        
    def call(self, x, encOutputs, mask=None, mask2=None):
        seqLen = tf.shape(x)[1]
        x = self.embd(x)+self.posEnc[:, :seqLen, :]
        for layer in self.decLayers:
            x = layer(x, encOutputs, mask, mask2)
        return self.dense(x)

enL = len(engVocab)
ptL = len(pt_vocab)
del engVocab
del pt_vocab

encInput = keras.layers.Input(shape=[None,])
decInput = keras.layers.Input(shape=[None,])

padMask = createPadMask(encInput)
decMask1 = createPadMask(decInput)
decMask2 = createMask(tf.shape(decInput)[1])
decMaskT = tf.maximum(decMask1, decMask2)
decMask3 = tf.maximum(padMask, decMask1)

Enc = Encoder(ptL)
encOut = Enc(encInput, padMask)


dec = Decoder(enL, enL+num_oov_buckets)
decOut = dec(decInput, encOut, decMaskT, padMask)

model = keras.models.Model(inputs=[encInput, decInput], outputs=decOut)

def lossfn(ytrue, ypred):
    loss_ = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(ytrue, ypred)
    mask = tf.cast(K.not_equal(ytrue, 0), loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/(tf.reduce_sum(mask) + 1e-7)


def myAcc(ytrue, ypred):
    a = tf.cast(tf.cast(tf.argmax(ypred, axis=-1), tf.float32) == ytrue, tf.float32)
    mask = tf.cast(K.not_equal(ytrue, 0), tf.float32)
    return tf.reduce_sum(a*mask)/tf.reduce_sum(mask)


optim = keras.optimizers.Adam(learning_rate=1e-5, clipvalue=1.0)
model.compile(loss=lossfn, optimizer=optim, metrics=[myAcc])
model.fit(data, epochs=3, validation_data=val_data)
