# -*- coding: utf-8 -*-

# generate causal data
import glob
import numpy as np
import pandas as pd
import os
import time
import re
import sys
import pickle
from datetime import date, timedelta
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
print(os.getcwd())
 
'''
python build_causal_evt_ICEWS.py IND 2015 2016 14 3 28
'''

try:
    COUNTRY = sys.argv[1] # e.g., IND
    STARTYEAR = sys.argv[2]
    ENDYEAR = sys.argv[3]
    OUTCOME_EVT = int(sys.argv[4]) # 14 is protest # see http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf
    DELTA = int(sys.argv[5]) # time granularity
    TOP = int(sys.argv[6])  # top ? locations
except:
    print('COUNTRY CODE, STARTYEAR, ENDYEAR, OUTCOME_EVT, DELTA, TOP')
    exit()

print('COUNTRY =',COUNTRY,'STARTYEAR=',STARTYEAR,'ENDYEAR=',ENDYEAR,'OUTCOME_EVT=',OUTCOME_EVT, 'DELTA=',DELTA,'TOP=',TOP)
'''
We have one json file containing events of a country in several years. (Make edits if your event file is different.)
E.g., icews_events_IND.json
'''
EVENT_FILE = 'your-data-path/icews_events_{}.json'.format(COUNTRY)

 
NEW_FOLDER = '../data/'
if TOP > 0:
    FILE_PREFIX = '{}-{}{}{}-{}-t{}/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT,TOP)
else:
    FILE_PREFIX = '{}-{}{}{}-{}/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT)
print('dataset',FILE_PREFIX)

int_start_year = int(STARTYEAR)
int_last_year = int(STARTYEAR)-1
str_start_year = str(STARTYEAR)
str_last_year = str(int(STARTYEAR)-1)
int_end_year = int(ENDYEAR)
str_end_year = str(ENDYEAR)

if not os.path.exists(NEW_FOLDER + FILE_PREFIX):
    os.makedirs(NEW_FOLDER + FILE_PREFIX)


def main():
    event_df = pd.read_json(EVENT_FILE, lines=True)
    event_df = event_df.loc[(event_df['Event Date']> str_last_year + '-12-28') & (event_df['Event Date'] <= str_end_year + '-12-31')][['Province', 'District', 'City',
       'Country', 'CAMEO Code', 'Event Date','Story ID','Event ID','Event Sentence']]
    print('event_df',len(event_df))
    event_df["loc"] = event_df["Province"]
    event_df["main"] = event_df["CAMEO Code"].apply(lambda x: get_cameo_main(x))
    if COUNTRY == 'IND': # Here, we fixed the target locations
        loc_list = ['Delhi', 'Maharashtra', 'Andhra Pradesh', 'Tamil Nadu',
        'Uttar Pradesh', 'Punjab', 'Jammu and Kashmir',  'Karnataka', 
        'Kerala', 'Bengal','Haryana', 'Bihar','Gujarat',   # should be West Bengal now
        'Madhya Pradesh',  'Rajasthan', 'Odisha', 'Chandigarh','Assam', 'Jharkhand','Himachal Pradesh', 
        'Uttarakhand','Chhattisgarh',  'Puducherry', 
        'Manipur', 'Telangana','Tripura',  'Meghalaya', 'Arunachal Pradesh'] # 28 use string contain
        if TOP > 0:
            loc_list = loc_list[:TOP]
    print('loc_list',len(loc_list))
     

    start_date = date(int_start_year, 1, 1)
    if int_end_year == 2017:
        end_date = date(2017, 3, 27)
    else:
        end_date = date(int_end_year, 12, 31)
    print(start_date,end_date)
    delta = timedelta(days=DELTA)

    outcome = []
    feature = []
    adjacency = []
    ii = 0
    while start_date <= end_date:
        start_date_str = start_date.strftime("%Y-%m-%d")
        last_date_str = (start_date - delta).strftime("%Y-%m-%d")
        next_date_str = (start_date + delta).strftime("%Y-%m-%d")
        if ii % 50 == 0:
            event_df = event_df.loc[event_df['Event Date'] >= last_date_str]
            print ('this day',start_date_str,time.ctime(),len(event_df))
        ii+=1
        outcome_date = []
        feature_date = []
        for loc in loc_list:
            if COUNTRY == 'IND':
                loc_df = event_df.loc[event_df["loc"].str.contains(loc, na=False)]
            elif COUNTRY in ['RUS','THA','EGY']:
                loc_df = event_df.loc[event_df["loc"].isin(loc)]
                
            last_day_df = loc_df.loc[(loc_df['Event Date'] >= last_date_str) & (loc_df['Event Date'] < start_date_str)]
            # last_day_event = list(last_day_df['CAMEO Code'].values)
            last_day_main_event = list(last_day_df['main'].values)
            
            this_day_df = loc_df.loc[(loc_df['Event Date'] >= start_date_str) & (loc_df['Event Date'] < next_date_str)]
            # this_day_event = list(this_day_df['CAMEO Code'].values)
            this_day_main_event = list(this_day_df['main'].values)
            
         
            # [outcome] how many of [OUTCOME_EVT] in code_list
            outcome_event = [v for v in this_day_main_event if v == OUTCOME_EVT]
            outcome_date.append(len(outcome_event))
            
            # [feature] tensor
            event_count_list = [0 for i in range(20)]
            for ev in last_day_main_event:
                event_count_list[ev-1] += 1
            feature_date.append(event_count_list)
             
        feat = sp.csr_matrix(np.array(feature_date))
        feature.append(feat)
        adj = cosine_similarity(feat)
        adjacency.append(sp.csr_matrix(np.array(adj)))
        
        outcome.append(outcome_date)
        
        start_date += delta

    # write feature
    file = NEW_FOLDER + FILE_PREFIX + 'feat-evt.pkl'
    with open(file,'wb') as f:
        pickle.dump(feature, f)
    print(file, 'write DONE')

    # write adj
    file = NEW_FOLDER + FILE_PREFIX + 'adj.pkl'
    with open(file,'wb') as f:
        pickle.dump(adjacency, f)
    print(file, 'write DONE')
    
    file = NEW_FOLDER + FILE_PREFIX + 'outcome.txt'
    write_matrix(file, outcome)

def write_matrix(file, r):
    f = open(file,'w') 
    for i in range(len(r)):
        for j in range(len(r[i])):
            f.write("{}".format(r[i][j]))
            if j < len(r[i])-1:
                f.write(',')
        f.write('\n')
    f.close()
    print(file, 'write DONE')

def get_cameo_main(code):
    code = int(code)
    if code < 100:
        return code // 10
    if code // 10 > 20:
        return code // 100
    else:
        return code // 10

main()
