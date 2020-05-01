#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:05:17 2020

@author: subha
"""

from time import clock, sleep
from pprint import pprint
import pandas as pd
from spacy.matcher import Matcher
from nltk.stem import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
from dateutil import relativedelta
import constants as cs
import utilities as utls
import parser  as par
import  pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta
#from .utils import loadDocumentIntoSpacy, countWords, loadDefaultNLP
from typing import *
from tika import parser
import matplotlib.pyplot as plt
import seaborn as sns
import io, os, subprocess, code, glob, re, traceback, sys, inspect
import json, re, pickle,logging, nltk ,spacy,string ,tika ,zipfile,csv
tika.TikaClientOnly = True
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import random  
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import en_core_web_sm
nlp = en_core_web_sm.load()
stop_words_ = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

######-----------------############--------------------------------------------------###############

# # #if not aleady download, plz load it
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# =============================================================================
# Solution approach and rationale:-
# Cross Industry Standard Process for Data Mining (CRISP–DM) framework.It involves a series of steps:
# 	1.Business understanding
# 	2.Data understanding
# 	3.Data Preparation & EDA
# 	4.Model Building
# 	5.Model Evaluation
#   6.Model Validation
# 	7.Model Deployment
# 
# =============================================================================
######### Business Understanding #############
#Giving each candiate a score against a JD and finding the top 10 based on content based recomendations
# and cosine similarity.

############# Data Understanding ###################

resume = pd.read_csv("resume.csv")
resume.head()
resume.info()
resume.isnull().sum()

############ Data Preparation  & EDA #####################
#Selecting the columns for the resume corpus.we will consider the columns: 'filename','total_exp' ,'skills', 'qualifications','work_exp','extras','weak_words','teamwork','action_words','metrics','leadership','result_driven','analytical','communication'
cols = ['filename','qualifications','total_exp','skills','work_exp','teamwork','communication','analytical','result_driven','leadership','metrics','action_words','weak_words']
resume = resume[cols]

# checking for the percenatge null values again.
round(resume.isnull().sum()/len(resume)*100,0)

#Let´s check the NA's by plotting them
x = resume.columns
y = resume.isnull().sum()
plt.figure(figsize=(10,6))
sns.set()
sns.barplot(x,y)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(),
            height + 2,
            str(100*round(int(height)/resume.shape[0],1)) +"",
            fontsize=10, ha='center', va='bottom')
ax.set_xlabel("Columns")
ax.set_ylabel("NA's (%)")
plt.xticks(rotation=80)
plt.show()

resume = resume.fillna("")
text_cols = ['qualifications','total_exp','skills','work_exp','teamwork','communication','analytical','result_driven','leadership','metrics','action_words','weak_words']
resume['text'] = resume[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
resume_text = resume[['filename','text']]
resume_text['text'] = resume_text['text'].apply(utls.normalize)
resume_text.head(5)

#------------- Reading Job Descriptions-----------------#

jd = pd.read_csv('jd.csv')
jd = jd[['title','desc']]
jd['desc'] = jd['desc'].apply(utls.normalize)
job_id = random.randint(0,len(jd) - 1) # Random job id against which we will provide  our recomendations.
jd.loc[job_id,"desc"]


####--------------------Top 10 Profile Recomendations based on job descriptions-----------------#########


def profile_recommendations(top,resume,scores):
  recommendation = pd.DataFrame(columns = ['JobID', 'Filename','Score'])
  count = 0
  for i in top:
      recommendation.at[count, 'JobID'] = job_id
      recommendation.at[count, 'Filename'] = resume['filename'][i]
      recommendation.at[count, 'Score'] =  scores[count]
      count += 1
  return recommendation

#------------------------------#----------------------------------------------------------------------

def process_resume_jd(doc_list, docs , doc_list_name,n):
# function to transform questions and display progress
    for doc in docs:
        doc_list.append(doc)
        if len(doc_list) % 10000 == 0:
            progress = len(doc_list)/n * 100
            print("{} is {}% complete.".format(doc_list_name, round(progress, 1)))
            
item_resume = []     
process_resume_jd(item_resume, resume_text.text, "Candidates Resumes", len(resume_text))
user_jd = []     
process_resume_jd(user_jd, jd.desc, "Job Descriptions",len(jd))

# Contains the processed questions for Doc2Vec
train_corpus = []

for i in range(len(resume_text)):
    # Question strings need to be separated into words
    # Each question needs a unique label
    train_corpus.append(TaggedDocument(item_resume[i].split(),resume_text.index))
    #tagged_words.append(TaggedDocument(docs_jd[i].split(),user_jd.loc[i,'title']))
    if i % 10000 == 0:
        progress = i/len(item_resume) * 100
        print("{}% complete".format(round(progress, 2)))
        
        

#------------------------------#----------------------------------------------------------------------#
        
model = Doc2Vec(dm = 1 , min_count= 2 , vector_size = 120,epochs = 63)
model.build_vocab(train_corpus)
model.train(train_corpus,total_examples=model.corpus_count,epochs=model.epochs)


item_resume_split = []
for text in item_resume:
    item_resume_split.append(text.split())
    
user_jd_split = []
for text in user_jd:
   user_jd_split.append(text.split())
    

model.most_similar('design')
model.most_similar('sql')

     
# Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(user_jd_split) - 1)
inferred_vector = model.infer_vector(user_jd_split[job_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))


# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(job_id, ' '.join(user_jd_split[job_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
#recomendations of using doc2vec and cosine similarity
    
#doc2vec_recomendations = profile_recommendations(top, resume, scores)

#-------------------TFIDF Implementation of cosine similarity ang getting the top 10 resume ------#

tfidf_vectorizer = TfidfVectorizer()
#fitting and transforming the resume vector
tfidf_resume = tfidf_vectorizer.fit_transform(resume_text['text']) 
tfidf_resume

#Cretating the job/user Corpus let's take the  jd.
jd_user_q = jd.iloc[[job_id]]
#fitting and transforming the job vector
tfidf_jd = tfidf_vectorizer.transform(jd_user_q['text'])
cos_similarity_tfidf = map(lambda x: cosine_similarity(tfidf_jd, x),tfidf_resume)
output = list(cos_similarity_tfidf)

#recomendations of using TFIDF and cosine similarity

top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:5]
list_scores = [output[i][0][0] for i in top]
profile_recommendations(top,resume, list_scores)


#-----------------K Nearest Neighboor recomendations usinf TF IDF vector-----------------------------------------------------###################

from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import GridSearchCV

#hyper parameter tuning
model = NearestNeighbors()
#Making models with hyper parameters sets
model_knn = NearestNeighbors(n_neighbors = 11, radius=1.0,
                 algorithm='brute', leaf_size = 30, metric='minkowski',
                 p=2, metric_params=None, n_jobs= -1)
model_knn.fit(tfidf_resume)
NNs = model_knn.kneighbors(tfidf_jd, return_distance= True)
top = NNs[1][0][1:]
index_score = NNs[0][0][1:]
profile_recommendations(top, resume, index_score)













