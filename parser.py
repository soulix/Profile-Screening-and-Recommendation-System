#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:26:38 2020
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
import  pandas as pd
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta
#from .utils import loadDocumentIntoSpacy, countWords, loadDefaultNLP
from typing import *
from tika import parser
import io, os, subprocess, code, glob, re, traceback, sys, inspect
import json, re, pickle,logging, nltk ,spacy,string ,tika ,zipfile,csv
tika.TikaClientOnly = True
nlp = spacy.load('en_core_web_sm')

#------- pdf and docx resume and jd reading -------#
#In entire code "text" means normal str object and "nlp_text" means spacy doc object
#resume and jd reading from doc and pdf files using tika, before that server need to host
#into localhost through 9998 port.Please find the instruction.txt file to host it into local env.
# Glob module matches certain patterns
url = 'http://localhost:9998/'
doc_files = glob.glob("/Users/mahika/Documents/21/resumes/*.doc",recursive=True)
docx_files = glob.glob("/Users/mahika/Documents/21/resumes/*.docx",recursive=True)
pdf_files = glob.glob("/Users/mahika/Documents/21/resumes/*.pdf",recursive=True)
rtf_files = glob.glob("/Users/mahika/Documents/21/resumes/*.rtf",recursive=True)
text_files = glob.glob("/Users/mahika/Documents/21/resumes/*.txt",recursive=True)
files = set(doc_files + docx_files + pdf_files + rtf_files + text_files)
files = list(files)
print ("%d files identified" %len(files))

csv_file = "resume.csv"
csv_columns = ['fileName', 'name', 'email','phone_number',
                'academics','qualifications','work_exp','total_experience',
                'extras','teamwork','communication','analytical','result_driven',
                'leadership','metrics','action_words','weak_words']
    
information = []

for f in files:
    info = {}
    text =  str(parser.from_file(f,url)["content"]).strip()
    nlp_text = nlp(text)      
    info['fileName'] = os.path.basename(f)   
    info['name'] = utls.findName(nlp_text,f)
    info['email'] = utls.extract_email_addresses(text)    
    info['phone_number'] = utls.fetch_phone(text)    
    info['academics'] = sum(utls.extract_academics(text).values(), [])    
    info['qualifications'] = utls.find_qualifications(info['academics'],text)[0]
    info['work_exp'] = sum(utls.extract_experience(text).values(), [])    
    info['total_experience'] = utls.calculate_experience(text) 
    info['extras'] = sum(utls.extract_extras(text).values(), [])    
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
    info['extras'] = ",".join(str(e) for e in info['extras'])
    info['work_exp'] = ",".join(str(e) for e in info['work_exp'])
    info['academics'] = ",".join(str(e) for e in info['academics'])
    info['email'] = ",".join(str(e) for e in info['email'])
    info['qualifications'] = ",".join(str(e) for e in info['qualifications'])
    print (info)
    information.append(info)



def export_to_csv(dict):   
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in information:
                writer.writerow(data)
    except IOError:
        print("I/O error")



files  = glob.glob("/Users/mahika/Documents/21/*.docx",recursive=True)


for f in files:
    info = {}
    text =  str(parser.from_file(f,url)["content"]).strip()
    nlp_text = nlp(text)      
    info['fileName'] = os.path.basename(f)   
    info['name'] = utls.findName(nlp_text,f)
    info['email'] = utls.extract_email_addresses(text)    
    info['phone_number'] = utls.fetch_phone(text)    
    info['academics'] = sum(utls.extract_academics(text).values(), [])    
    info['qualifications'] = utls.find_qualifications(info['academics'],text)[0]
    info['work_exp'] = sum(utls.extract_experience(text).values(), [])    
    info['total_experience'] = utls.calculate_experience(text) 
    info['extras'] = sum(utls.extract_extras(text).values(), [])    
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
    info['extras'] = ",".join(str(e) for e in info['extras'])
    info['work_exp'] = ",".join(str(e) for e in info['work_exp'])
    info['academics'] = ",".join(str(e) for e in info['academics'])
    info['email'] = ",".join(str(e) for e in info['email'])
    info['qualifications'] = ",".join(str(e) for e in info['qualifications'])
    print (info)
    information.append(info)




export_to_csv(information)


    
 













