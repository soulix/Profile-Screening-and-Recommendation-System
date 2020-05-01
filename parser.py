#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:26:38 2020
@author: subha
"""

# from time import clock, sleep
# from pprint import pprint
# from spacy.matcher import Matcher
# from nltk.stem import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from datetime import datetime
# from dateutil import relativedelta
import constants as cs
import utilities as utls
import  pandas as pd
#from .utils import loadDocumentIntoSpacy, countWords, loadDefaultNLP
# from typing import *
from tika import parser
import io, os, subprocess, code, glob, re, traceback, sys, inspect
import json, re, pickle,logging, nltk ,spacy,string ,tika ,zipfile,csv
tika.TikaClientOnly = True
import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en_core_web_sm')
url = 'http://localhost:9998/'

#------- pdf and docx resume and jd reading -------#
#In entire code "text" means normal str object and "nlp_text" means spacy doc object
#resume and jd reading from doc and pdf files using tika, before that server need to host
#into localhost through 9998 port.Please find the instruction.txt file to host it into local env.
# Glob module matches certain patterns

doc_files = glob.glob("/Users/mahika/Documents/21/CVs/*.doc",recursive=True)
docx_files = glob.glob("/Users/mahika/Documents/21/CVs/*.docx",recursive=True)
pdf_files = glob.glob("/Users/mahika/Documents/21/CVs/*.pdf",recursive=True)
rtf_files = glob.glob("/Users/mahika/Documents/21/CVs/*.rtf",recursive=True)
text_files = glob.glob("/Users/mahika/Documents/21/CVs/*.txt",recursive=True)
files = set(doc_files + docx_files + pdf_files + rtf_files + text_files)
files = list(files)

#files  =  glob.glob("/Users/mahika/Documents/21/dataset/*.pdf",recursive=True)
print ("%d files identified" %len(files))

# x = list(skills.columns.values)
#using my secondary data set to list out skilss in data science and web application developemnt domain.
skills = pd.read_csv('final_skills.csv')
skills = skills.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
skills = skills.drop_duplicates().values.tolist()
skills = utls.skill_catalog(skills)

information = []


for f in files:
    info = {}
    text =  str(parser.from_file(f,url)["content"]).strip()
    nlp_text = nlp(text)      
    info['filename'] = os.path.basename(f)  
    info['phone_number'] = utls.fetch_phone(text)
    info['email'] = utls.extract_email_addresses(text)            
    info['academics'] = sum(utls.extract_academics(text).values(), [])
    info['qualifications'] = utls.find_qualifications(info['academics'],text)[0]
    info['total_exp'] = utls.calculate_experience(text)
    info['skills'] = utls.find_skills(nlp_text,skills)
    work_exp = sum(utls.extract_experience(text).values(), [])  
    extras   = sum(utls.extract_extras(text).values(), [])
    info['work_exp'] = utls.getUniqueWords(work_exp + extras)  
    competencies = utls.find_competencies(text) 
    info['teamwork']      = competencies.get('teamwork',"")    
    info['communication'] = competencies.get('communication',"")    
    info['analytical']    = competencies.get('analytical',"")    
    info['result_driven'] = competencies.get('result_driven',"")    
    info['leadership']    = competencies.get('leadership',"")   
    info['metrics']       = competencies.get('metrics',"")   
    info['action_words']  = competencies.get('action_words',"")   
    info['weak_words']    = competencies.get('weak_words',"")
    #converting list to string for further data mining.
    info['weak_words'] = ",".join(str(e) for e in info['weak_words']) 
    info['teamwork'] = ",".join(str(e) for e in info['teamwork']) 
    info['action_words'] = ",".join(str(e) for e in info['action_words']) 
    info['metrics'] = ",".join(str(e) for e in info['metrics']) 
    info['leadership'] = ",".join(str(e) for e in info['leadership']) 
    info['result_driven'] = ",".join(str(e) for e in info['result_driven']) 
    info['analytical'] = ",".join(str(e) for e in info['analytical'])
    info['communication'] = ",".join(str(e) for e in info['communication']) 
    info['qualifications'] = ",".join(str(e) for e in info['qualifications'])
    info['work_exp'] = ",".join(str(e) for e in info['work_exp'])
    info['skills'] = ",".join(str(e) for e in info['skills'])    
    print (info)
    information.append(info)

utls.export_to_csv(information)









############################################################################################



















































