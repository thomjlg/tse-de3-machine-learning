#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:13:51 2022

@author: thomasjaulgey - enzosana
"""

#importation des modules nécessaires
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


#Si erreur à la compilatiuon, faire l'installation de stopwords avec:
#    import nltk
#    nltk.download('stopwords')



#On définit la taille du vocabulaire
vocab_size = 5000
#on définit la dimension d'intégration
embedding_dim = 64
#On définit la longueur max, ici 200
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .7


desc = []
cate = []

#ouverture du fichier !!Lien a modifier en fonction de votre répertoire!
with open('/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TP/TP-DeepLearning/data/dataset_to_model.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    #On parcourt le fichier ligne par ligne
    for row in reader:
        cate.append(row[0])
        article = row[1]
        #On supprime les stopwords du fichier source tels que "un", "est", "le",...
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        desc.append(article)


#On définit la taille du jeu de train
train_size = int(len(desc) * training_portion)

#On definit le jeu train_desc de 0 à la taille du jeu de train
#Idem pour train_cate
train_desc = desc[0: train_size]
train_cate = cate[0: train_size]

#On définit les jeux de validation desc et cate
validation_desc = desc[train_size:]
validation_cate = cate[train_size:]


#On vectorise le corpus de texte
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_desc)
word_index = tokenizer.word_index

#On créé un dictionnaire 
dict(list(word_index.items())[0:10])

#On utilise la fonction texts_to_sequences pour transformer une chaîne de texte en une liste
train_sequences = tokenizer.texts_to_sequences(train_desc)

#On transforme la liste en un tableau 2D avec la fonction pad_sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#On réalise la même chose que ci-dessus sur les données validation
validation_sequences = tokenizer.texts_to_sequences(validation_desc)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#On vectorise les lables
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(cate)

#On crée un tableau avec les vecteurs label qu'on transforme en tableau 2D
training_cate_seq = np.array(label_tokenizer.texts_to_sequences(train_cate))
validation_cate_seq = np.array(label_tokenizer.texts_to_sequences(validation_cate))

#On crée un dictionnaire
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


#On crée une fonction pour retrouver le texte original
def decode_desc(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#on crée notre model par keras
model = tf.keras.Sequential([
    # On ajoute une couche d'intégration avec un vocabulaire d'entrée de taille 5000 et une dimension d'intégration de sortie de taille 64,  définie plus haut
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    
    # On utilise la fonction d'activation ReLU
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # On ajoute une couche avec 6 unité et la fonction d'activation Softmax
    # Si nous avons plusieurs sorties, softmax convertit les couches de sorties en une distribution de probabilité.
    tf.keras.layers.Dense(6, activation='softmax')
])

#On affiche un résumé du model
model.summary()

#On configure le model avec le loss et les metrics
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#On définit le nombre d'epoch, ici 10
num_epochs = 10
#on fit notre model (= on l'entraine) 
history = model.fit(train_padded, training_cate_seq, epochs=num_epochs, validation_data=(validation_padded, validation_cate_seq), verbose=2)







