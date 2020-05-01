#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:33:13 2020

@author: subha
"""


# import string
# import pandas as pd
#from spacy.matcher import Matcher
from nltk.stem import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from datetime import datetime
#from dateutil import relativedelta
from collections import Counter
import constants as cs
#from collections import defaultdict
from datetime import date
import io, os, subprocess, code, glob, re, traceback, sys, inspect
import json, re, pickle,logging, nltk ,spacy,string ,tika ,zipfile,csv
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words_ = set(stopwords.words('english'))
import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en_core_web_sm')
wordnet_lemmatizer = WordNetLemmatizer()


def string_found(string1, string2):
    if re.search(r"\b" + re.escape(string1) + r"\b", string2):
        return True
    return False

#Extracting full name
def findName(doc, filename):
        """
        Helper function to extract name from nlp doc
        :param doc: SpaCy Doc of text
        :param filename: used as backup if NE cannot be found
        :return: str:NAME_PATTERN if found, else None
        """
        to_chain = False
        all_names = []
        person_name = None

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if not to_chain:
                    person_name = ent.text.strip()
                    to_chain = True
                else:
                    person_name = person_name + " " + ent.text.strip()
            elif ent.label_ != "PERSON":
                if to_chain:
                    all_names.append(person_name)
                    person_name = None
                    to_chain = False
        if all_names:
            return all_names[0]
        else:
            try:
                base_name_wo_ex = os.path.splitext(os.path.basename(filename))[0]
                return base_name_wo_ex + " (from filename)"
            except:
                return None



#Function to extract Email address from a string object using regular expressions
def extract_email_addresses(text):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    # using set() to remove duplicated  from list 
    email = list(set(r.findall(text)))
    return email

#Function to extract Phone Numbers from string object using regular expressions
"""
Utility function that fetches phone number in the resume.
Params: resume_text type: string
returns: phone number type:string
"""
def fetch_phone(resume_text):
    regular_expression = re.compile(cs.GET_PHONE(3, 3, 10),re.IGNORECASE)
    result = re.search(regular_expression, resume_text)
    phone = ''
    if result:
      result = result.group()
      for part in result:
        if part:
          phone += part
    if phone == '':
      for i in range(1, 10):
        for j in range(1, 10-i):
          regular_expression =re.compile(cs.GET_PHONE(i, j,10), re.IGNORECASE)
          result = re.search(regular_expression, resume_text)
          if result:
            result = result.groups()
            for part in result:
              if part:
                phone += part
          if phone != '':
            return phone
    return phone
    
def findCity(doc):
        counter = Counter()
        """
        Helper function to extract most likely City/Country from nlp doc
        :param doc: SpaCy Doc of text
        :return: str:city/country if found, else None
        """
        for ent in doc.ents:
            if ent.label_ == "GPE":
                counter[ent.text] += 1

        if len(counter) >= 1:
            return counter.most_common(1)[0][0]
        return None


def calculate_experience(resume_text):
    
  try:
    experience = 0
    start_month = -1
    start_year = -1
    end_month = -1
    end_year = -1
    regular_expression = re.compile(cs.date_range, re.IGNORECASE)
    regex_result = re.search(regular_expression, resume_text)
    while regex_result:
      date_range = regex_result.group()
      year_regex = re.compile(cs.year)
      year_result = re.search(year_regex, date_range)
      if (start_year == -1) or (int(year_result.group()) <= start_year):
        start_year = int(year_result.group())
        month_regex = re.compile(cs.months_short, re.IGNORECASE)
        month_result = re.search(month_regex, date_range)
        if month_result:
          current_month = cs.get_month_index(month_result.group())
          if (start_month == -1) or (current_month < start_month):
            start_month = current_month
      if date_range.lower().find('present' or 'current') != -1:
        end_month = date.today().month # current month
        end_year = date.today().year # current year
      else:
        year_result = re.search(year_regex, date_range[year_result.end():])
        if (end_year == -1) or (int(year_result.group()) >= end_year):
          end_year = int(year_result.group())
          month_regex = re.compile(cs.months_short, re.IGNORECASE)
          month_result = re.search(month_regex, date_range)
          if month_result:
            current_month = cs.get_month_index(month_result.group())
            if (end_month == -1) or (current_month > end_month):
              end_month = current_month
      resume_text = resume_text[regex_result.end():]
      regex_result = re.search(regular_expression, resume_text)

    return end_year - start_year  # Use the obtained month attribute
  except Exception as exception_instance:
    logging.error('Issue calculating experience: '+str(exception_instance))
    return None

"""
Utility function that cleans the resume_text.
Params: resume_text type: string
returns: cleaned text ready for processing
"""
## initialise the inbuilt Stemmer and the Lemmatizer




'''
Helper function to extract all the raw text from sections of
resume specifically for professionals/freshers
:param text: Raw text of resume
:return: dictionary of entities
'''

def extract_academics(text):

    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.SIMILAR_TO['ACADEMICS'])
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.SIMILAR_TO['ACADEMICS']:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities


def extract_experience(text):

    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.SIMILAR_TO['EXPERINCE'])
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.SIMILAR_TO['EXPERINCE']:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities



def extract_extras(text):

    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.SIMILAR_TO['EXTRA'])
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.SIMILAR_TO['EXTRA']:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities



"""
Utility function that fetches degree and degree-info from the resume.
Params: resume_text Type: string
returns:
degree Type: List of strings
info Type: List of strings
"""
### if education section is null then search with entire text in fetch_qualification
def fetch_qualifications(resume_text):
    qualifications = cs.EDUCATION
    degree = []
    info = []
    for qualification in qualifications:
      qual_regex = r'[^a-zA-Z]'+qualification+r'[^a-zA-Z]'
      regular_expression = re.compile(qual_regex, re.IGNORECASE)
      regex_result = re.search(regular_expression, resume_text)
      while regex_result:
        degree.append(qualification)
        resume_text = resume_text[regex_result.end():]
        lines = [line.rstrip().lstrip()
        for line in resume_text.split('\n') if line.rstrip().lstrip()]
        if lines:
          info.append(lines[0])
        regex_result = re.search(regular_expression, resume_text)
    return degree, info



#### qualifications exception when there is no education present into resume:education sections
def find_qualifications(academics,text):
     if len(academics) != 0 :
         qualifications =  fetch_qualifications(str(academics))
         if len(qualifications) == 0:
             qualifications = fetch_qualifications(text)            
         return(qualifications)
     else:
         qualifications = fetch_qualifications(text)
         return(qualifications)
        



def find_competencies(text):
    '''
    Helper function to extract competencies from resume text
    :param resume_text: Plain resume text
    :return: dictionary of competencies
    '''
    competency_dict = {}

    for competency in cs.COMPETENCIES.keys():
        for item in cs.COMPETENCIES[competency]:
            if string_found(item,text):
                if competency not in competency_dict.keys():
                    competency_dict[competency] = [item]
                else:
                    competency_dict[competency].append(item)
    
    return competency_dict



def countWords(line: str) -> int:
    """
    Counts the numbers of words in a line
    :param line: line to count
    :return count: num of lines
    """
    count = 0
    is_space = False
    for c in line:
        is_not_char = not c.isspace()
        if is_space and is_not_char:
            count += 1
        is_space = not is_not_char
    return count


def skill_catalog(skills):
    skills = ','.join(map(str, skills))
    skills = skills.split(",")
    removeable = str.maketrans('', '', "[]''")
    skills = [s.translate(removeable) for s in skills]
    skills = [x.strip(' ') for x in skills]
    skills = getUniqueWords(skills)
    return skills



def position_catalog(position):
    position = position.str.lower().drop_duplicates().values.tolist()
    position = ','.join(map(str, position))
    position = position.split(",")
    removeable = str.maketrans('', '', "[]''")
    position = [s.translate(removeable) for s in position]
    position = [x.strip(' ') for x in position]
    position = getUniqueWords(position)
    return position



def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords


def find_skills(doc,skills):
        """
        Helper function to extract skills from spacy nlp text
        :param doc: object of `spacy.tokens.doc.Doc`
        :return: list of skills extracted
        """
        tokens = [token.text for token in doc if not token.is_stop]
        skillset = []
        # check for one-grams
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)
        # check for bi-grams and tri-grams
        for token in doc.noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
        return [i.capitalize() for i in set([i.lower() for i in skillset])]



def n2w(n):
    try:
      return (cs.num2words[n])
    except KeyError:
         try:
            return (cs.num2words[n-n%10] + cs.num2words[n%10].lower())
         except KeyError:
             return ('Number out of range')
             
             
             


def export_to_csv(dict):   
    try:
        with open(cs.csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames= cs.csv_columns)
            writer.writeheader()
            for data in dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")
 
def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets .
    Every dataset is lower cased  
    """
    string = re.sub("'", "",string)
    string = re.sub("(\\d|\\W)+"," ",string) 
    string = string.replace("nbsp", "")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string)  
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    string = string.lower()
    return string.strip()
  
def normalize(text):
  clean_text = []
  clean_text_two = []
  text = clean_str(text)
  clean_text = [ wordnet_lemmatizer.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
  clean_text_two = [word for word in clean_text if black_txt(word)]
  return " ".join(clean_text_two)


             

           
