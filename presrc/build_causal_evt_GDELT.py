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
python build_causal_evt_GDELT.py CA 2015 2020 14 1 13 0
'''
try:
    COUNTRY = sys.argv[1] # e.g., EG
    STARTYEAR = sys.argv[2]
    ENDYEAR = sys.argv[3]
    OUTCOME_EVT = int(sys.argv[4]) # 14 is protest # see http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf
    DELTA = int(sys.argv[5])
    TOP = int(sys.argv[6]) # top ? locations
    ROOT = int(sys.argv[7]) # 1 means selecting only the main event reported by a news report
except:
    print('COUNTRY CODE, STARTYEAR, ENDYEAR, OUTCOME_EVT, DELTA, TOP, ROOT')
    exit()

print('COUNTRY =',COUNTRY,'STARTYEAR=',STARTYEAR,'ENDYEAR=',ENDYEAR,'OUTCOME_EVT=',OUTCOME_EVT, 'DELTA=',DELTA,'TOP=',TOP)

'''
You might have several json files containing events of a country in different years.
E.g., event.2015.CA.json, ... event.2020.CA.json
'''
filepath = 'your-data-path/event.*.{}.json'.format(COUNTRY)
EVENT_FILE_LIST = glob.glob(filepath)

NEW_FOLDER = '../data/'
if ROOT:
    if TOP < 0:
        FILE_PREFIX = '{}-{}{}{}-{}-r/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT)
    else:
        FILE_PREFIX = '{}-{}{}{}-{}-t{}-r/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT,TOP)
else:
    if TOP < 0:
        FILE_PREFIX = '{}-{}{}{}-{}/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT)
    else:
        FILE_PREFIX = '{}-{}{}{}-{}-t{}/'.format(COUNTRY, str(STARTYEAR)[2:],str(ENDYEAR)[2:],DELTA, OUTCOME_EVT,TOP)
print('dataset',FILE_PREFIX)

int_start_year = int(STARTYEAR)
int_last_year = int(STARTYEAR)-1
str_start_year = str(STARTYEAR)
str_last_year = str(int(STARTYEAR)-1)
int_end_year = int(ENDYEAR)
str_end_year = str(ENDYEAR)

if not os.path.exists(NEW_FOLDER + FILE_PREFIX):
    os.makedirs(NEW_FOLDER + FILE_PREFIX)

# get events
def get_events():
    frames = []
    for f in EVENT_FILE_LIST:
        year = f.split('.')[1]
        if int(year) < int_last_year:
            continue
        if int(year) > int_end_year:
            continue
        print(year)
        subdf = pd.read_json(f, lines=True)
        frames.append(subdf)
    event_df = pd.concat(frames)
    print('event_df len =',len(event_df))
    event_df = event_df.sort_values(by='event_date', ascending=True)
    event_df = event_df[['ActionGeo_Fullname', 'IsRootEvent', 'event_date','EventCode','MentionIdentifier']]
    event_df['event_date']=event_df.event_date.astype('str')
    return event_df

def get_loc_info(event_df):
    event_df['loc'] = event_df.ActionGeo_Fullname.apply(lambda x: x.split(', ')[::-1])
    print(event_df['loc'])
    try:
        event_loc_df = pd.DataFrame(event_df['loc'].to_list(), columns=['country','state','city','empty'])
    except:
        event_loc_df = pd.DataFrame(event_df['loc'].to_list(), columns=['country','state','city'])
    print(event_loc_df,'event_loc_df len =',len(event_loc_df))
    event_df.reset_index(drop=True, inplace=True)
    event_loc_df.reset_index(drop=True, inplace=True)
    event_df = pd.concat([event_df, event_loc_df], axis=1)
    loc_name = 'state'
    event_df = event_df.dropna(subset=[loc_name])
    print('event_df len =',len(event_df), ' no',loc_name)
    return event_df
 

def main():
    event_df = get_events()
    if ROOT == 1:
        event_df = event_df.loc[event_df['IsRootEvent'] > 0]
        print('event_df len =',len(event_df), 'rooted')

    event_df["main"] = event_df["EventCode"].apply(lambda x: get_cameo_main(x))
    event_df = event_df.dropna(subset=['main'])
    event_df['main'] = event_df['main'].astype(int)
    print('event_df len =',len(event_df), ' drop error code')
    print(event_df[['main']])
    
    event_df = get_loc_info(event_df)
    loc_name = 'state'
    if COUNTRY == 'CA':
        loc_list = [v for v in event_df[loc_name].unique() if v != "Canada (general)"]
        all_locs = loc_list
        print(all_locs)
    elif COUNTRY == 'AS':
        loc_list = [v for v in event_df[loc_name].unique() if v != ""]
        all_locs = loc_list
        print(all_locs,'all_locs')
        print(loc_list,'loc_list')
    elif COUNTRY == 'NI':
        loc_list = [
            ["Abuja Federal Capital Territory","Benue","Kogi","Kwara","Niger","Plateau"], # North Central
            ["Enugu","Anambra","Imo","Abia","Ebonyi"], # South East
            ["Borno","Yobe","Bauchi","Adamawa","Gombe","Taraba"], # North East
            ["Rivers","Delta","Edo","Cross River","Bayelsa","Akwa Ibom"], # South South
            ["Kano","Kaduna","Sokoto","Zamfara","Katsina","Jigawa","Kebbi"], # North West
            ["Lagos","Oyo","Ogun","Ekiti","Ondo","Osun"] # South West
        ]
        all_locs = [item for sublist in loc_list for item in sublist]
    else:
        tmp_df = event_df.groupby([loc_name]).size().reset_index().rename(columns={0:'count'})
        tmp_df.sort_values(by='count', ascending=False, inplace=True)
        print(tmp_df,'tmp_df')
        sorted_locs = list(tmp_df[loc_name].values)
        sorted_locs = [item for item in sorted_locs if item.strip() !='']
        loc_list = sorted_locs[:TOP]
        all_locs = loc_list
    
    print('loc_list',len(loc_list),'all_locs',len(all_locs))
    event_df = event_df.loc[event_df[loc_name].isin(all_locs)]
    print('event_df len =',len(event_df), ' for selected',loc_name)
    available_urls = event_df['MentionIdentifier'].unique()
    print('available_urls len =',len(available_urls))

    file = NEW_FOLDER + FILE_PREFIX + 'location.txt'
    with open (file, 'w') as f:
        f.write("\n".join(all_locs))

    start_date = date(int_start_year, 1, 1)
    end_date = date(int_end_year, 12, 31)
    print(start_date,end_date)
    delta = timedelta(days=DELTA)
    print(event_df[['IsRootEvent','event_date','EventCode','main']])

    outcome = []
    feature = []
    adjacency = []

    ii = 0
    while start_date <= end_date:
        start_date_str = start_date.strftime("%Y%m%d")
        last_date_str = (start_date - delta).strftime("%Y%m%d")
        next_date_str = (start_date + delta).strftime("%Y%m%d")
        if ii % 100 == 0:
            event_df = event_df.loc[event_df['event_date'] >= last_date_str]
            print ('this day',start_date_str,time.ctime(),len(event_df))
        ii+=1
        outcome_date = []
        feature_date = []

        for i_loc in range(len(loc_list)):
            loc = loc_list[i_loc]
            if COUNTRY in ['EG','NI']:
                loc_df = event_df.loc[event_df[loc_name].isin(loc)]
            else:
                loc_df = event_df.loc[event_df[loc_name] == loc]
                
            last_day_df = loc_df.loc[(loc_df['event_date'] >= last_date_str) & (loc_df['event_date'] < start_date_str)]
            # last_day_event = list(last_day_df['EventCode'].values)
            last_day_main_event = list(last_day_df['main'].values)
            
            this_day_df = loc_df.loc[(loc_df['event_date'] >= start_date_str) & (loc_df['event_date'] < next_date_str)]
            # this_day_event = list(this_day_df['EventCode'].values)
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
    try:
        code = int(code)
        if code < 100:
            return code // 10
        if code // 10 > 20:
            return code // 100
        else:
            return code // 10
    except:
        return np.nan

main()
