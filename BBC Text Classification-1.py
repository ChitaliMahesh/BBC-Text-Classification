#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df=pd.read_csv(r"C:\Users\mahes\Downloads\archive (5)\bbc-text.csv")
df.head()


# In[5]:


category=pd.get_dummies(df.category)
new_df=pd.concat([df,category],axis=1)
new_df=new_df.drop(columns='category')
new_df


# In[7]:


# convert dataframe to numoy array

texts = new_df['text'].values
label = new_df[['business', 'entertainment', 'politics', 'sport', 'tech']].values


# # Bagi train set dan test set

from sklearn.model_selection import train_test_split

train_texts,test_texts,train_labels,test_labels=train_test_split(texts,label,test_size=0.3)


# In[10]:


# Stopword removal and Porter stemming

import numpy as np
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

porterStemmer = PorterStemmer()

fixed_train_texts = []
for text in train_texts:
  removed_text = remove_stopwords(text)   # Stopword removal
  removed_text = porterStemmer.stem_sentence(removed_text) # Port steemming
  fixed_train_texts.append(removed_text)

fixed_train_texts = np.array(fixed_train_texts, dtype='O')   # Train set 

fixed_test_texts = []
for text in test_texts:
  removed_text = remove_stopwords(text)
  removed_text = porterStemmer.stem_sentence(removed_text) # Port steemming
  fixed_test_texts.append(removed_text)


# In[11]:


print(f'text ({len(train_texts[0])}): {train_texts[0]}')
print(f'removed ({len(fixed_train_texts[0])}): {fixed_train_texts[0]} ({len(fixed_train_texts[0])})')


# In[12]:


# Text  Tokenization

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(fixed_train_texts)
tokenizer.fit_on_texts(fixed_test_texts)
print(tokenizer.word_index)


# In[15]:


# sample sequence

train_sequences = tokenizer.texts_to_sequences(fixed_train_texts)
test_sequences = tokenizer.texts_to_sequences(fixed_test_texts)


# In[19]:


# Sequence padding

padded_train = pad_sequences(train_sequences)
padded_test = pad_sequences(test_sequences)


# In[24]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Embedding,Dropout,LSTM


# In[26]:


model=Sequential()

model.add(Embedding(input_dim=5000,output_dim=16))
model.add(LSTM(64))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))


# In[27]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[28]:


model.summary()


# In[30]:


history=model.fit(padded_train,train_labels,epochs=30,batch_size=128,verbose=2,validation_data=[padded_test,test_labels])


# In[31]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.show


# In[32]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show


# In[ ]:




