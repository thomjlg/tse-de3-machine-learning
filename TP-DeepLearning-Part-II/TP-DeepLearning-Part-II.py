#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:13:51 2022

@author: thomasjaulgey - enzosana
"""

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .7


#df = pd.read_csv('/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TP/TP-DeepLearning/data/dataset_to_model.csv')
#X = df['description_pre'].values
#Y = df['Category'].values


desc = []
cate = []

with open('/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TP/TP-DeepLearning/data/dataset_to_model.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        cate.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        desc.append(article)



train_size = int(len(desc) * training_portion)

train_desc = desc[0: train_size]
train_cate = cate[0: train_size]

validation_desc = desc[train_size:]
validation_cate = cate[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_desc)
word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])

train_sequences = tokenizer.texts_to_sequences(train_desc)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


validation_sequences = tokenizer.texts_to_sequences(validation_desc)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(cate)

training_cate_seq = np.array(label_tokenizer.texts_to_sequences(train_cate))
validation_cate_seq = np.array(label_tokenizer.texts_to_sequences(validation_cate))

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_desc(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 10
history = model.fit(train_padded, training_cate_seq, epochs=num_epochs, validation_data=(validation_padded, validation_cate_seq), verbose=2)







