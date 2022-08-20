#!/usr/bin/env python
# coding: utf-8

# In[20]:


import spacy
from nltk.tag import pos_tag,pos_tag_sents
import regex as re
from nltk.tokenize import word_tokenize
import pickle
import pandas as pd
nlp = spacy.load('en_core_web_sm')
def make_input(test):
    test = test.lower()
    string =  pos_tag(word_tokenize(test), tagset='universal')
    #print(string)
    pos = []
    ent = []
    doc = nlp(test)
    for element in doc.ents:
        ent.append(element.label_)
    ent = " ".join(ent)
    for word in string:
        #print(pos)
        pos.append(word[1])
    pos = " ".join(pos)
   #print(pos)
    data = [[test,pos,ent]]
    test_df = pd.DataFrame(data,columns=["ColumnA",'tag',"entity"])
    return test_df


# In[10]:


from sklearn.base import BaseEstimator, TransformerMixin
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# In[11]:


loaded_model = pickle.load(open("finalModel.sav", 'rb'))


# In[73]:


loaded_model.predict(make_input("""21 rue Pierre Motte

SAINT-DENIS, Île-de-France(IL), 97400
"""))


# In[ ]:




