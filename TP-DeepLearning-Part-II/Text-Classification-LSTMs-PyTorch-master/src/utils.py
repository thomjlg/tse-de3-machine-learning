
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

# tuto  '/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TD/TP-DeepLearning-Part-II/Text-Classification-LSTMs-PyTorch-master/data/tweets.csv'
# perso '/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TD/TP-DeepLearning/data/dataset_to_model.csv'
class Preprocessing:
	
	def __init__(self, args):
		self.data = '/Users/thomasjaulgey/Documents/_TSE_IUT/_FISA_DE/_DE3/_DATA/_Apprentissage_Automatique/_TD/TP-DeepLearning/data/dataset_to_model.csv'
		self.max_len = args.max_len
		self.max_words = args.max_words
		self.test_size = args.test_size
		print('max len', self.max_len)
		print('max words', self.max_words)
		print('test size', self.test_size)
		
	def load_data(self):
		df = pd.read_csv(self.data)
		#df.drop(['id','keyword','location'], axis=1, inplace=True)
		
		X = df['description_pre'].values
		Y = df['Category'].values
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size)
		print('x_train', self.x_train)
		print('y_train', self.y_train)
		print('x_test', self.x_test)
		print('y_test', self.y_test)
		
	def prepare_tokens(self):
		self.tokens = Tokenizer(num_words=self.max_words)
		self.tokens.fit_on_texts(self.x_train)

	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return sequence.pad_sequences(sequences, maxlen=self.max_len)