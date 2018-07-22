# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import read_dataset
import svm_rank
import random
import rank_measures
import kDPP
import numpy.linalg as la



def sample_kDPP(Y,k):
    X=np.array(Y)
    return kDPP.kDPPGreedySample(X,k)
    
    
def normalize_sd(nrecords):
    n=len(nrecords)
    d=len(nrecords[0])
    for j in range(d):
        sum=0.0
        for i in range(n):
            sum+=nrecords[i][j]
        mean=sum/n
        var=0.0
        for i in range(n):
            nrecords[i][j]-=mean
            var+=nrecords[i][j]**2
        std_d=np.sqrt(var/n)
        for i in range(n):
            nrecords[i][j]/=std_d
            
def normalize(nrecords):
    n=len(nrecords)
    d=len(nrecords[0])
    for j in range(d):
        mx=0.0
        for i in range(n):
            mx=max(mx,abs(nrecords[i][j]))
        for i in range(n):
            nrecords[i][j]/=mx
    
 
def regularize_data(X,delta):
    n=X.shape[0]
    K=np.dot(X,X.T)
    Kp=K+delta*np.eye(n)
    #(u,s,v)=la.svd(Kp)
    #print(s)
    L=la.cholesky(K+delta*np.eye(n))
    return L
    
def remove_zero_cols(nrecords):
    n=len(nrecords)
    d=len(nrecords[0])
    new_nrecords=[[] for e in range(n)]
    for i in range(d):
        mx=0.0
        for j in range(n):
            mx=max(mx,nrecords[j][i])
        if mx>0:
            for j in range(n):
                new_nrecords[j].append(nrecords[j][i])
    return new_nrecords

def text_fields_to_features(tf):
    (lower,upper)=(3,500)
    bag=[]
    for s in tf:
        g="".join([ c if c.isalpha() else " " for c in s ])
        g=g.lower()
        l=map(str.strip,g.split())
        l=filter(lambda x: len(x)>1,l)
        bag.extend(l)
    take=[]
    ubag=list(set(bag))
    for e in ubag:
        ec=bag.count(e)
        if lower<=ec and ec<=upper:
            take.append(e)
    take.sort()
    features=[]
    for s in tf:
        g="".join([ c if c.isalpha() else " " for c in s ])
        g=g.lower()
        l=map(str.strip,g.split())
        v=[]
        for i in range(len(take)):
            if take[i] in l: v.append(1.0)
            else: v.append(0.0)
        features.append(v)
    print('text features: %d'%len(take))     
    return features

def data_to_vectors(data,types,head,columns_to_keep):
    d=len(data[0])
    ndata=[[] for e in data]
    for c in range(len(data[0])):
        if head[c] not in columns_to_keep: continue
        if types[c]=='text':
            text_fields=[r[c] for r in data]
            text_features=text_fields_to_features(text_fields)
            for i in range(len(data)):
                ndata[i].extend(text_features[i])
        elif types[c]=='num':
            min_val=0        
            for i in range(len(data)):
                min_val=min(min_val,data[i][c])
            for i in range(len(data)):
                x=data[i][c]
                if min_val==-1: 
                    x+=1
                ndata[i].append(float(x))
        else:
            sval=set()
            for i in range(len(data)):
                sval.add(data[i][c])
            sval=list(sval)
            vals=len(sval)
            for i in range(len(data)):
                indicator=[0.0]*vals
                for k in range(vals):
                    if sval[k]==data[i][c]:
                        indicator[k]=1.0
                ndata[i]=ndata[i]+indicator
    normalize(ndata)
    print(len(ndata))
    print(len(ndata[0]))
    return ndata
    
'''    
(head,types,records)=read_dataset.read_dataset()   

num_score=head.index('decile_score')
ranks=[]
for e in records:
    ranks.append(e[num_score])
cols_no_ranks=head[:]
cols_no_ranks.remove('decile_score')
nrecords=data_to_vectors(records,types,head,cols_no_ranks)
print(len(nrecords))
#random.shuffle(nrecords)
size=100
#nrecords=nrecords[:500]
#ranks=ranks[:500]

ra=svm_rank.generate_ranking(nrecords[:size*5],ranks[:size*5],nrecords[-size:],ranks[-size:])
#ra=svm_rank.generate_ranking(nrecords[:size*2],ranks[:size*2],nrecords[:size*2],ranks[:size*2])
print(len(ra))

print(rank_measures.avg_swaps(ranks[-size:],ra))
random.shuffle(ra)
print(rank_measures.avg_swaps(ranks[-size:],ra))
'''

