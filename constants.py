#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:41:01 2020

@author: subha
"""
import nltk
from nltk.corpus import stopwords

# # #if not aleady download, plz load it
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
STOPWORDS = set(stopwords.words('english'))
# Constants
# LINES_FRONT = 3
# LINES_BACK = 3



csv_file = "resume.csv"
csv_columns = ['filename','phone_number', 'email','academics','qualifications',
               'total_exp','skills','work_exp','teamwork','communication',
               'analytical','result_driven','leadership','metrics','action_words','weak_words']
        

num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}


NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
# Regular expressinos used
bullet = r"\(cid:\d{0,2}\)"
#Education (Upper Case Mandatory)
EDUCATION         = [
                    'BE','B.E.', 'B.E', 'BS', 'B.S', 'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'BTECH', 'MTECH', 
                    'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII' , 'Bachelor of Arts','Bachelor of Science',
                    'Bachelor of Technology','Master of Technology','Master of Science','B.Tech.','B.Tech',
                    'M.Tech.','M.Tech',  'B.Sc','M.Sc','BA','MA','B.A','M.A', 'Bachelor of Commerce', 'B.Com'
                    'Master of Commerce','M.Com', 'B.Com.','M.Com.','Doctor of Philosophy','PhD','Ph.D.',
                    'DPhil','PHD','Ph.D' ,'Ed.D.', 'MCA','Master of Computer Applications',
                    'BCA','Bachelor of Computer Applications', 'MBA'
                    ]

NOT_ALPHA_NUMERIC = r'[^a-zA-Z\d]'
NUMBER            = r'\d+'
not_alpha_numeric = r'[^a-zA-Z\d]'
number = r'\d+'

  # regex explanation in order:
  # optional braces open
  # optional +
  # one to three digit optional international code
  # optional braces close
  # optional whitespace separator
  # i digits
  # optional whitespace separator
  # j digits
  # optional whitespace separator
  # n-i-j digits
def GET_PHONE(i,j,n):
  return r"\(?(\+)?(\d{1,3})?\)?[\s-]{0,1}?(\d{"+str(i)+"})[\s\.-]{0,1}(\d{"+str(j)+"})[\s\.-]{0,1}(\d{"+str(n-i-j)+"})"

#getting month index
def get_month_index(month):
    month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    return month_dict[month.lower()]

# For finding date ranges
MONTHS_SHORT      = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
MONTHS_LONG       = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
MONTH             = r'(' + MONTHS_SHORT + r'|' + MONTHS_LONG + r')'
YEAR              = r'(((20|19)(\d{2})))'



# there should be 1 non digit, followed by a whitespace
# then pin and trailing whitespace.
# This is to avoid phone numbers being read as pincodes
pincode = r"[^\d]"+not_alpha_numeric+"(\d{6})"+not_alpha_numeric
# For finding date ranges
months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
month = r'('+months_short+r'|'+months_long+r')'
year = r'((20|19)(\d{2})|(\d{2}))'
start_date = month+not_alpha_numeric+r"?"+year
end_date = r'(('+month+not_alpha_numeric+r"?"+year+r')|(present))'+not_alpha_numeric
longer_year = r"((20|19)(\d{2}))"
year_range = longer_year+not_alpha_numeric+r"{1,3}"+longer_year
date_range =  r"("+start_date+not_alpha_numeric+r"{1,3}"+end_date+r")|("+year_range+r")"




SIMILAR_TO = {
    'ACADEMICS' : [ 'studies',
                    'institute', 
                    'school', 
                    'college',
                    'university',
                    'education',
                    'qualifications',
                    'academics'
                ],
    
    'EXPERINCE' : [ 'experience',
                    'work',
                    'projects',
                    'professional experience',
                    'employment history',
                    'duties',
                    'internship', 
                    'training',
                    'research',
                    'roles',
                    'responsibility', 
                    'conference',
                    'work experience'
                ],
    
    'SKILLS' : [    'skills',
                    'key skills',
                    'languages', 
                    'technology', 
                    'framework', 
                    'tools', 
                    'certifications',
                    'database',
                    'proficiency',
                    'Technology Stack'
            ],
    
    'EXTRA' : [     'achievements',
                    'additional', 
                    'awards',
                    'miscellaneous', 
                    'publications',
                    'others',
                    'accolades',
                    'interests'
            ]
}


COMPETENCIES = {
    'teamwork': [
        'supervised',
        'facilitated',
        'planned',
        'plan',
        'served',
        'serve',
        'project lead',
        'managing',
        'managed',
        'lead ',
        'project team',
        'team',
        'conducted',
        'worked',
        'gathered',
        'organized',
        'mentored',
        'assist',
        'review',
        'help',
        'involve',
        'share',
        'support',
        'coordinate',
        'cooperate',
        'contributed'
    ],
    'communication': [
        'addressed',
        'collaborated',
        'conveyed',
        'enlivened',
        'instructed',
        'performed',
        'presented',
        'spoke',
        'trained',
        'author',
        'communicate',
        'define',
        'influence',
        'negotiated',
        'outline',
        'proposed',
        'persuaded',
        'edit',
        'interviewed',
        'summarize',
        'translate',
        'write',
        'wrote',
        'project plan',
        'business case',
        'proposal',
        'writeup'
    ],
    'analytical': [
        'process improvement',
        'competitive analysis',
        'aligned',
        'strategive planning',
        'cost savings',
        'researched ',
        'identified',
        'created',
        'led',
        'measure',
        'program',
        'quantify',
        'forecasr',
        'estimate',
        'analyzed',
        'survey',
        'reduced',
        'cut cost',
        'conserved',
        'budget',
        'balanced',
        'allocate',
        'adjust',
        'lauched',
        'hired',
        'spedup',
        'speedup',
        'ran',
        'run',
        'enchanced',
        'developed'
    ],
    'result_driven': [
        'cut',
        'decrease',
        'eliminate',
        'increase',
        'lower',
        'maximize',
        'rasie',
        'reduce',
        'accelerate',
        'accomplish',
        'advance',
        'boost',
        'change',
        'improve',
        'saved',
        'save',
        'solve',
        'solved',
        'upgrade',
        'fix',
        'fixed',
        'correct',
        'achieve'           
    ],
    'leadership': [
        'advise',
        'coach',
        'guide',
        'influence',
        'inspire',
        'instruct',
        'teach',
        'authorized',
        'chair',
        'control',
        'establish',
        'execute',
        'hire',
        'multi-task',
        'oversee',
        'navigate',
        'prioritize',
        'approve',
        'administer',
        'preside',
        'enforce',
        'delegate',
        'coordinate',
        'streamlined',
        'produce',
        'review',
        'supervise',
        'terminate',
        'found',
        'set up',
        'spearhead',
        'originate',
        'innovate',
        'implement',
        'design',
        'launch',
        'pioneer',
        'institute'
    ],
     'metrics': [
        'saved',
        'increased',
        '$ ',
        '%',
        'percent',
        'upgraded',
        'fundraised ',
        'millions',
        'thousands',
        'hundreds',
        'reduced annual expenses ',
        'profits',
        'growth',
        'sales',
        'volume',
        'revenue',
        'reduce cost',
        'cut cost',
        'forecast',
        'increase in page views',
        'user engagement',
        'donations',
        'number of cases closed',
        'customer ratings',
        'client retention',
        'tickets closed',
        'response time',
        'average',
        'reduced customer complaints',
        'managed budget',
        'numeric_value'
    ],
    'action_words': [
        'developed',
        'led',
        'analyzed',
        'collaborated',
        'conducted',
        'performed',
        'recruited',
        'improved',
        'founded',
        'transformed',
        'composed',
        'conceived',
        'designed',
        'devised',
        'established',
        'generated',
        'implemented',
        'initiated',
        'instituted',
        'introduced',
        'launched',
        'opened',
        'originated',
        'pioneered',
        'planned',
        'prepared',
        'produced',
        'promoted',
        'started',
        'released',
        'administered',
        'assigned',
        'chaired',
        'consolidated',
        'contracted',
        'co-ordinated',
        'delegated',
        'directed',
        'evaluated',
        'executed',
        'organized',
        'oversaw',
        'prioritized',
        'recommended',
        'reorganized',
        'reviewed',
        'scheduled',
        'supervised',
        'guided',
        'advised',
        'coached',
        'demonstrated',
        'illustrated',
        'presented',
        'taught',
        'trained',
        'mentored',
        'spearheaded',
        'authored',
        'accelerated',
        'achieved',
        'allocated',
        'completed',
        'awarded',
        'persuaded',
        'revamped',
        'influenced',
        'assessed',
        'clarified',
        'counseled',
        'diagnosed',
        'educated',
        'facilitated',
        'familiarized',
        'motivated',
        'participated',
        'provided',
        'referred',
        'rehabilitated',
        'reinforced',
        'represented',
        'moderated',
        'verified',
        'adapted',
        'coordinated',
        'enabled',
        'encouraged',
        'explained',
        'informed',
        'instructed',
        'lectured',
        'stimulated',
        'classified',
        'collated',
        'defined',
        'forecasted',
        'identified',
        'interviewed',
        'investigated',
        'researched',
        'tested',
        'traced',
        'interpreted',
        'uncovered',
        'collected',
        'critiqued',
        'examined',
        'extracted',
        'inspected',
        'inspired',
        'summarized',
        'surveyed',
        'systemized',
        'arranged',
        'budgeted',
        'controlled',
        'eliminated',
        'itemised',
        'modernised',
        'operated',
        'organised',
        'processed',
        'redesigned',
        'reduced',
        'refined',
        'resolved',
        'revised',
        'simplified',
        'solved',
        'streamlined',
        'appraised',
        'audited',
        'balanced',
        'calculated',
        'computed',
        'projected',
        'restructured',
        'modelled',
        'customized',
        'fashioned',
        'integrated',
        'proved',
        'revitalized',
        'set up',
        'shaped',
        'structured',
        'tabulated',
        'validated',
        'approved',
        'catalogued',
        'compiled',
        'dispatched',
        'filed',
        'monitored',
        'ordered',
        'purchased',
        'recorded',
        'retrieved',
        'screened',
        'specified',
        'systematized',
        'conceptualized',
        'brainstomed',
        'tasked',
        'supported',
        'proposed',
        'boosted',
        'earned',
        'negotiated',
        'navigated',
        'updated',
        'utilized'
    ],
    'weak_words': [
        'i',
        'got',
        'i\'ve',
        'because',
        'our',
        'me',
        'he',
        'her',
        'him',
        'she',
        'helped',
        'familiar',
        'asssisted',
        'like',
        'enjoy',
        'love',
        'did',
        'tried',
        'attempted',
        'worked',
        'approximately',
        'managed',
        'manage',
        'create',
        'created'
    ]
}


