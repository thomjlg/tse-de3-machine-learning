#!/usr/bin/env python
# coding: utf-8

# # Deep learning - DE3 final Lab (Evaluation) 
# 
# This project belongs to the NLP domain. The task is straightforward: assign the correct job category to a job description. This is thus a multi-class classification task with 28 classes to choose from.
# The data has been retrieved from CommonCrawl. The latter has been famously used to train OpenAI's GPT-3 model. The data is therefore representative of what can be found on the English speaking part of the Internet. The goal of this project is to design a solution that accurate to predict the job based on the job descriptions. 
# 
# ## Evaluation
# 
# First of all, solutions are evaluated according to the Macro F1 metric, The Macro F1 score is simply the arithmetic average of the F1 score for each class.
# 
# ## Datasets
# 
# **data.json**
# Contains job descriptions as well as genders for the training set, which contains 217,197 samples. If you're using pandas, then you can easily open this with pd.read_json.
# 
# **label.csv**
# Contains job labels for the training set.
# 
# **categories_string.csv**
# Provides a mapping between job labels and label integers

# Import modules
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

# ## Baseline 1 (no deep learning)
# For this part we propose to train and test a straightforward logistic regression model on the tf-idf vectors extracted from the corpus. We are not applying any particular pre-processing to start and will progressively try to reduce the dimension of the vocabulary.
# Read all three files to obtain three dataframes (pandas)

# read train.json
df = pd.read_json('data/train.json')
# read label
label = pd.read_csv('data/label.csv')
# read categories_string
categories = pd.read_csv('data/categories_string.csv')

# Display the 3 first rows of each dataframe
df.head(5)
label.head(3)
categories.head(3)

# Add a column in df based on the column Category from the label dataframe
#df['label'] = label['Category']
df = pd.merge(df, label, on='Id')
df.head(3)

# Visualize the distribution of the gender attribute and the Category one using appropriate graphics (barplot)
# Example of visualisation libraries: seaborn, matplotlib etc. 
import seaborn as sns
import matplotlib.pyplot as plt
graph_gender = sns.countplot(x="gender", data=df)
plt.show()
graph_category = sns.countplot(x="Category", data=df)
plt.show()


# **What do you observe ?**
# Your answer: Données très inégales, cela risque de fausser le modèle, mais en moyenne ça sera ok
# For the following part, let's focus on the top-5 jobs (category). 
# Create a new dataframe based on `df` but that contains observations that belongs to the top-5 most frequent jobs. 
top_values = df['Category'].value_counts()
liste = top_values.nlargest(5).index.tolist()
print(liste)
df_top5 = df[df['Category'].isin(liste)]

df_top5.sample(10)

# Lets convert the text to lower case to reduce the size of the vocabulary
df_top5["description_lower"] = [x.lower() for x in df_top5.description]
df_top5.sample(5)

# Split the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(df_top5["description_lower"], 
                                                    df_top5["Category"], test_size=0.33, random_state=42)

# Convert the text to tf-idf vectors 
transformer = TfidfVectorizer()
transformer.fit(X_train.values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(X_train.values)
X_test = transformer.transform(X_test.values)

# Check the shape of X_train and y_train
print("X_train shape is : ", X_train.shape)
print("y_train shape is : ", y_train.shape)

# Fit a logistic regression model on X_train with 2000 max iterations
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Check the macro f1-score
test_f1 = f1_score(y_test, y_pred, average='macro')
print('Macro F1:  %0.3f' % test_f1)


# ## What about pre-processing ? 
# Let's try basic pre-processing for textual data. 
# - removing stop words
# - lemmatization 
# - stemmatization 
# To proceed we can use the `nltk` library for instance (other alternatives : `sklearn`, `spacy`)

# Import nltk and the list of stop words
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

# Take a look at the list
print(list(stop))

# Remove stop words from df 
df_top5['description_wo_stop'] = df_top5["description_lower"].apply(
    lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))

X_train, X_test, y_train, y_test = train_test_split(df_top5["description_wo_stop"], 
                                                    df_top5["Category"], test_size=0.33, random_state=42)

# Tf-idf vectorization
transformer = TfidfVectorizer(stop_words=list(stop)).fit(X_train.values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(X_train.values)
X_test = transformer.transform(X_test.values)

# Logistic regression - Same as before
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_f1 = f1_score(y_test, y_pred, average='macro')
print('Macro F1:  %0.3f' % test_f1)

# **What do you observe ?**
# Your answer : mêmes performances du modèle mais on a moins de variables, donc potentiel gain de temps de calcul sans dégrader les performances du modèle

# Let add some more pre-processing. 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import re

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    # Lowercase text
    sentence = sentence.lower()
    # Remove whitespace
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    # Remove weblinks
    rem_url=re.sub(r'http\S+', '',cleantext)
    # Remove numbers
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

# **Describe the different pre-processing methods used by the `preprocess` function**
# Your answer: 
# - texte en minuscule
# - suppressions des espaces en trop
# - suppression des liens web
# - suppression des nombres
# - suppression des mots de moins de 2 lettres et des stop words de plus de 2 lettres pour ne garder que le texte important

# Let's apply the function to our data
df_top5['description_pre']=df_top5['description'].map(lambda s:preprocess(s)) 

# Following the same steps as before, split the data into a train and a test set. They learn a logistic regression and print the f1 score obtained. 

X_train, X_test, y_train, y_test = train_test_split(df_top5["description_pre"], 
                                                    df_top5["Category"], test_size=0.33, random_state=42)

# Tf-idf vectorization
transformer = TfidfVectorizer(stop_words=list(stop)).fit(X_train.values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(X_train.values)
X_test = transformer.transform(X_test.values)

# Logistic regression - Same as before
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_f1 = f1_score(y_test, y_pred, average='macro')
print('Macro F1:  %0.3f' % test_f1)


# ### Export dataset top5
to_export = df_top5[['Category', 'description_pre']]
to_export.to_csv('data/dataset_to_model.csv', index=False)

# **What do you observe ?**
# Your answer: score identique aux précédent
# Let's try to tune the regularisation parameter of our model. 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# ## Let's dive into Deep Learning
# ### Transformers and attention mechanisms
# The paper ‘Attention Is All You Need’ describes transformers and what is called a 
# sequence-to-sequence architecture. Sequence-to-Sequence (or Seq2Seq) is a neural net that transforms
# a given sequence of elements, such as the sequence of words in a sentence, into another sequence. 
# 
# Seq2Seq models are particularly good at translation, where the sequence of words from one language is transformed into a sequence of different words in another language. A popular choice for this type of model is Long-Short-Term-Memory (LSTM)-based models. With sequence-dependent data, the LSTM modules can give meaning to the sequence while remembering (or forgetting) the parts it finds important (or unimportant). Sentences, for example, are sequence-dependent since the order of the words is crucial for understanding the sentence. LSTM are a natural choice for this type of data.
# 
# Seq2Seq models consist of an Encoder and a Decoder. The Encoder takes the input sequence and maps it into a higher dimensional space (n-dimensional vector). That abstract vector is fed into the Decoder which turns it into an output sequence. The output sequence can be in another language, symbols, a copy of the input, etc. To solve the task of document classification we will use the pre-trained model **Bert**. 
# 
# #### TO DO 
# 
# 1. As a first exercice, you will have to read the paper **Attention Is All You Need**, to get a general understanding of transformers. You can also look for additional resources online (tutorial, video etc.) To keep things clear I recommend to focus on transformers for NLP (text). The paper can be found [here](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 
# 
# **This step is mandatory as you will need to unserstand the basics of transformers for what will follow. In addition, you will have questions specifically related to this architecture during your last evaluation.**
# 
# 2. We will then use a pre-trained model based usin the popular [Huggin Face](https://huggingface.co/docs/transformers/index) for pytorch. More precisely, we plan on using the [Bert](https://huggingface.co/docs/transformers/model_doc/bert) model. The following cells guide you toward using Bert. On top of applying the model to our data, you will have to answer a few questions along the project (including the ones above). These **questions are mandatory**. 
# 
# 3. Finally, the last step is the fine-tuning. Generally speaking, fine tuning a model refers to re-training the last layers of a deep architecture on your data. Again, you will be guided on how to proceed. 

# **Lets start !**
# The first thing that you need to do is to install the transformers librariy 
# 
# `import sys
# !{sys.executable} -m pip install transformers`

# Import the library
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch 
from transformers import BertTokenizer

# Let's take a look at some popular pre-trained models available in Huggingface. 
# 
# 1. We start with the model Camembert [Camembert](https://camembert-model.fr/) trained on french corpus by Facebook and Inria teams. 
# 2. GPT2 a pre-trained model for text generation
# 3. Bert !
# 
# **Note:** You can also directly try the online [demo](https://transformer.huggingface.co/) of the library.

# Example 1 - Camembert

model_name = "camembert-base"# try also distilbert-base-cased for english
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)


all_sequences = [f"Jacques Chirac est un {tokenizer.mask_token}", 
            f"Antoine Griezman est un {tokenizer.mask_token}",
            f"Le camembert, c'est {tokenizer.mask_token}"]

for sequence in all_sequences:
    input = tokenizer.encode(sequence, return_tensors="pt")
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    for token in top_5_tokens:
        print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

