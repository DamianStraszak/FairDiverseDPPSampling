# -*- coding: utf-8 -*-
import numpy as np
import sys
from datetime import datetime

def days_between(s1,s2):
    if (len(s1)>12):
        date_format='%Y-%m-%d %H:%M:%S'
    else:
        date_format='%Y-%m-%d'
    d1 = datetime.strptime(s1,date_format)
    d2 = datetime.strptime(s2,date_format)
        
    return abs((d1-d2).days)
    

def days_data(s1):
    if (len(s1)>12):
        date_format='%Y-%m-%d %H:%M:%S'
    else:
        date_format='%Y-%m-%d'
    d1 = datetime.strptime(s1,date_format)
    return 1.0*(d1-datetime(1970, 1, 1)).days
 

charge_features={} 
    
def read_charge_features():
    global charge_features
    charge_features={} 
    f=open('charges_features.txt')
    s=f.readline()
    headers=map(str.strip,s.split(','))[1:]
    while True:
        l=f.readline()
        if l=='': break
        l=l.split(',')
        text=l[0]
        v=map(float,l[1:])
        d={}
        for i in range(len(v)):
            d[headers[i]]=v[i]
        charge_features[text]=d
    f.close()
    
    
def clean_record(r,nrs,header):
    v=[]
    d={}
    d['sex']=r[nrs['sex']]
    d['age']=r[nrs['age']]
    d['race']=r[nrs['race']]
    d['juv_fel_count']=int(r[nrs['juv_fel_count']])
    d['decile_score']=int(r[nrs['decile_score']])
    d['juv_misd_count']=int(r[nrs['juv_misd_count']])
    d['juv_other_count']=int(r[nrs['juv_other_count']])
    d['priors_count']=int(r[nrs['priors_count']])
    days=days_between(r[nrs['c_jail_in']],r[nrs['c_jail_out']])
    d['days_in_jail']=days
    d['c_charge_degree']=r[nrs['c_charge_degree']]
    d['is_recid']=r[nrs['is_recid']]
    d['is_violent_recid']=r[nrs['is_violent_recid']]
    cd=charge_features[r[nrs['c_charge_desc']]]
    d['c_offense_date']=days_data(r[nrs['c_offense_date']])
    for e in cd:
        d[e]=cd[e]
    if (int(r[nrs['decile_score']])<=0):
        raise NameError('score not specified')
    if len(header)!=len(d):
        raise NameError('lengths do not match')
    v=[d[e] for e in header]

    return v
    
def get_cols():
    header=[]
    type=[]
    header.append('sex')
    type.append('cat')
    header.append('age')
    type.append('num')
    header.append('race')
    type.append('cat')
    header.append('juv_fel_count')
    type.append('num')
    header.append('decile_score')
    type.append('num')
    header.append('juv_misd_count')
    type.append('num')
    header.append('juv_other_count')
    type.append('num')
    header.append('priors_count')
    type.append('num')
    header.append('days_in_jail')
    type.append('num')
    header.append('c_charge_degree')
    type.append('cat')
    header.append('is_recid')
    type.append('num')
    header.append('is_violent_recid')
    type.append('num')
    header.append('is_violent')
    type.append('num')
    header.append('is_drug')
    type.append('num')
    header.append('is_firearm')
    type.append('num')
    header.append('is_MinorInvolved')
    type.append('num')
    header.append('is_roadSafetyHazard')
    type.append('num')
    header.append('is_sexoffense')
    type.append('num')
    header.append('is_fraud')
    type.append('num')
    header.append('is_petty')
    type.append('num')
    header.append('c_offense_date')
    type.append('num')
    return header,type
    
    

        


# c_charge_desc -> violent /non-violent
#['sex', 'age', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', diff('c_jail_in', 'c_jail_out'),  diff('c_offense_date', 'c_arrest_date')  'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_charge_degree', diff('r_days_from_arrest', 'r_offense_date'), 'r_charge_desc', diff('r_jail_in','r_jail_out'), 'is_violent_recid', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc', 'v_decile_score']
    

    
    
    
def read_dataset():
    read_charge_features()    
    file_name="compas-scores.csv"
    f=open(file_name)
    data=[]
    line_no=0
    cols=0
    
    while True:
        line_no+=1
        l=f.readline()
        if l=='': break
        s=l.split(',')
        if line_no==1:
            cols=len(s)
        for e in s: e.strip()
        if len(s)==cols:
            data.append(s)
    print(len(data))
    
    filtered_data=[]
    header=[]
    nrs={}
    for i in range(len(data[0])):
        nrs[data[0][i]]=i
    head,types=get_cols()
    records=[]
    for r in data[1:]:
        try:
            records.append(clean_record(r,nrs,head))
        except Exception as e:
            #print(e)
            pass
            
            
    print(len(records))
    return (head,types,records)
    
  

 

#print(charge_features['Agg Abuse Elderlly/Disabled Adults'])    

















