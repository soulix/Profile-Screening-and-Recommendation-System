#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:26:38 2020

@author: subha
"""


import io, os, subprocess, code, glob, re, traceback, sys, inspect
import json, re, pickle,logging, nltk ,spacy,string ,tika ,zipfile
from time import clock, sleep
from pprint import pprint
import pandas as pd
from tika import parser
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
import re
import pandas as pd
import os
import sys

from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta
#from .utils import loadDocumentIntoSpacy, countWords, loadDefaultNLP
from typing import *

#------- pdf and docx resume and jd reading -------#
#In entire code "text" means normal str object and "nlp_text" means spacy doc object
#resume and jd reading from doc and pdf files using tika
# Glob module matches certain patterns
doc_files = glob.glob("/Users/mahika/Documents/21/resumes/*.doc",recursive=True)
docx_files = glob.glob("/Users/mahika/Documents/21/resumes/*.docx",recursive=True)
pdf_files = glob.glob("/Users/mahika/Documents/21/resumes/*.pdf",recursive=True)
rtf_files = glob.glob("/Users/mahika/Documents/21/resumes/*.rtf",recursive=True)
text_files = glob.glob("/Users/mahika/Documents/21/resumes/*.txt",recursive=True)
nlp = spacy.load('en_core_web_sm')
files = set(doc_files + docx_files + pdf_files + rtf_files + text_files)
files = list(files)
print ("%d files identified" %len(files))




information=[]

for f in files:
    print("Reading File %s"%f)
    #info is a dictionary that stores all the data obtained from parsing
    info = {}
    raw_text = parser.from_file(f)
    info['fileName'] = f
    text =  str(raw_text["content"]).strip()
    nlp_text = nlp(text)         
    info['name'] = utls.extract_full_name(nlp_text)
    info['email'] = utls.extract_email_addresses(text)
    info['phone_number'] = utls.fetch_phone(text)
    resume_entity_sections = utls.extract_entity_sections(text)
    info['qualifications'] = utls.fetch_qualifications_update(resume_entity_sections,text)
    info['experience'] = utls.work_experience(resume_entity_sections)
    info['total_experience'] = utls.get_total_experience(info['experience'])
    info['competencies'] = utls.extract_competencies(text,info['experience']) 
    info['measurable_results'] = utls.extract_measurable_results(text,info['experience'])
    print (info)
    information.append(info)



tika.initVM() 
raw_text = parser.from_file('Subhajit_Data_Science.docx')
print(raw_text["metadata"])
print(raw_text["content"].strip())  
text = str(raw_text['content']).strip() 
text = text.strip()
print(text)

#text = utls.clean_text(text)
#Extracting Entities and Preprocessing resume

nlp_text = nlp(text)
print(nlp_text)



name = utls.findName(nlp_text,filename)
email = utls.extract_email_addresses(text)
phone_number = utls.fetch_phone(text)
resume_entity_sections = utls.extract_entity_sections(text)
qualifications = utls.fetch_qualifications_update(resume_entity_sections,text)
experience = utls.work_experience(resume_entity_sections)
total_experience = utls.calculate_experience(text)
competencies = utls.extract_competencies(text,experience) 
measurable_results = utls.extract_measurable_results(text,experience)


filename  = "/Users/mahika/Documents/21/resumes/Subhajit_Data_Science.docx"

WORDS_LIST = {
    "Work": ["(Work|WORK)", "(Experience(s?)|EXPERIENCE(S?))", "(History|HISTORY)"],
    "Education": ["(Education|EDUCATION)", "(Qualifications|QUALIFICATIONS)"],
    "Skills": [
        "(Skills|SKILLS)",
        "(Proficiency|PROFICIENCY)",
        "LANGUAGE",
        "CERTIFICATION",
    ],
    "Projects": ["(Projects|PROJECTS)"],
    "Activities": ["(Leadership|LEADERSHIP)", "(Activities|ACTIVITIES)"],
}

cat = extractCategories(text)
wk_edu = findWorkAndEducation(cat,nlp_text,text,name)

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


 def extractCategories(text) -> Dict[str, List[Tuple[int, int]]]:
        """
        Helper function to extract categories like EDUCATION and EXPERIENCE from text
        :param text: text
        :return: Dict[str, List[Tuple[int, int]]]: {category: list((size_of_category, page_count))}
        """
        data = defaultdict(list)
        page_count = 0
        prev_count = 0
        prev_line = None
        prev_k = None
        for line in text.split("\n"):
            line = re.sub(r"\s+?", " ", line).strip()
            for (k, wl) in WORDS_LIST.items():
                # for each word in the list
                for w in wl:
                    # if category has not been found and not a very long line
                    # - long line likely not a category
                    if countWords(line) < 10:
                        match = re.findall(w, line)
                        if match:
                            size = page_count - prev_count
                            # append previous
                            if prev_k is not None:
                                data[prev_k].append((size, prev_count, prev_line))
                            prev_count = page_count
                            prev_k = k
                            prev_line = line
            page_count += 1

        # last item
        if prev_k is not None:
            size = page_count - prev_count - 1 # -1 cuz page_count += 1 on prev line
            data[prev_k].append((size, prev_count, prev_line))

        # choose the biggest category (reduce false positives)
        for k in data:
            if len(data[k]) >= 2:
                data[k] = [max(data[k], key=lambda x: x[0])]
        return data



def findWorkAndEducation(categories, doc, text, name) -> Dict[str, List[str]]:
        inv_data = {v[0][1]: (v[0][0], k) for k, v in categories.items()}
        line_count = 0
        exp_list = defaultdict(list)
        name = name.lower()

        current_line = None
        is_dot = False
        is_space = True
        continuation_sent = []
        first_line = None
        unique_char_regex = "[^\sA-Za-z0-9\.\/\(\)\,\-\|]+"

        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            match = re.search(r"^.*:", line)
            if match:
                line = line[match.end() :].strip()

            # get first non-space line for filtering since
            # sometimes it might be a page header
            if line and first_line is None:
                first_line = line

            # update line_countfirst since there are `continue`s below
            line_count += 1
            if (line_count - 1) in inv_data:
                current_line = inv_data[line_count - 1][1]
            # contains a full-blown state-machine for filtering stuff
            elif current_line == "Work":
                if line:
                    # if name is inside, skip
                    if name == line:
                        continue
                    # if like first line of resume, skip
                    if line == first_line:
                        continue
                    # check if it's not a list with some unique character as list bullet
                    has_dot = re.findall(unique_char_regex, line[:5])
                    # if last paragraph is a list item
                    if is_dot:
                        # if this paragraph is not a list item and the previous line is a space
                        if not has_dot and is_space:
                            if line[0].isupper() or re.findall(r"^\d+\.", line[:5]):
                                exp_list[current_line].append(line)
                                is_dot = False

                    else:
                        if not has_dot and (
                            line[0].isupper() or re.findall(r"^\d+\.", line[:5])
                        ):
                            exp_list[current_line].append(line)
                            is_dot = False
                    if has_dot:
                        is_dot = True
                    is_space = False
                else:
                    is_space = True
            elif current_line == "Education":
                if line:
                    # if not like first line
                    if line == first_line:
                        continue
                    line = re.sub(unique_char_regex, '', line[:5]) + line[5:]
                    if len(line) < 12:
                        continuation_sent.append(line)
                    else:
                        if continuation_sent:
                            continuation_sent.append(line)
                            line = " ".join(continuation_sent)
                            continuation_sent = []
                        exp_list[current_line].append(line)

        return exp_list

