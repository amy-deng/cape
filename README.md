# CAPE

## Preparing Data
**1. Obtain GDELT/ICEWS event JSON files.**

One example row (GDELT event):
```
{"GlobalEventID": "719023380", "event_date": 20171231, "Actor1Code": "nan", "Actor1Name": "nan", "Actor2Code": "GBR", "Actor2Name": "UNITED KINGDOM", "IsRootEvent": 0, "EventCode": "040", "QuadClass": 1, "GoldsteinScale": 1.0, "NumMentions": 1, "NumArticles": 1, "AvgTone": 5.297, "ActionGeo_Type": 1, "ActionGeo_Fullname": "Poland", "ActionGeo_CountryCode": "PL", "SOURCEURL": "http://www.elle.com/culture/a14524053/john-kennedy-jackie-kennedy-queen-elizabeth-meeting-buckingham-palace/", "SentenceID": 18, "MentionIdentifier": "http://www.elle.com/culture/a14524053/john-kennedy-jackie-kennedy-queen-elizabeth-meeting-buckingham-palace/"}
```
**2. Run python files in `presrc`**

Examples
```
python build_causal_evt_GDELT.py NI 2017 2018 14 1 6 0
```

```
python build_causal_evt_ICEWS.py IND 2015 2016 14 3 28
```

**3. Manually build binary adjacency matrix `geoadj.txt`**

## Individual Treatment Effect Estimation
**Run CAPE**
```
python train_causal.py --loop 10 -m cape_cau -d NI --i_t 1
```
**Run a baseline model**
```
python train_causal_baselines.py --loop 10 -m cfrmmd -d NI --i_t 1
```

### Event Forecasting with Causal information
**Run CAPE**
```
python train_event_with_causal.py --loop 10 -m cape -d NI 
```
**Add noise to data**
```
python train_event_with_causal.py --loop 10 -m cape -d NI --train_noise 0.1
```
