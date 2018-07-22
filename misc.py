import numpy as np
import numpy.linalg as la

def plogp(p):
    if p<0.0001:    return 0
    return -p*np.log(p)

def H_entropy(p):
    return plogp(p)+plogp(1-p)
    
    
def KL_dist_to_uniform(p):
    n=len(p)
    prob=1.0/n
    r=0.0
    for e in p:
        if e<0.0001: e=0.0001
        r+=prob*np.log(prob/e)
    return r    
    
def entropy_prob(p):
    r=0.0
    for e in p:
        r+=plogp(e)
    return r
    
def comp_fractions(S,label1,label2):
    p=[]
    for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
        r=0.0
        for e in S:
            if (label1[e]==i and label2[e]==j): r+=1.0/len(S)
        p.append(r)
    return p

    
def fraction_ones(l):
    sum=0.0
    for e in l:
        sum+=e
    return sum/len(l)

def log_diversity(Y):
    n=Y.shape[0]
    
    (sign,ldet)=la.slogdet(np.eye(n)*0.01+np.dot(Y,np.transpose(Y)))
    #(sign,ldet)=la.slogdet(np.dot(Y,np.transpose(Y)))
    return ldet
    
    
def clean_file(path):
    f=open(path,'w')
    f.close()


def write_to_csv(path,items):
    f=open(path,'a')
    s=str(items[0])
    for e in items[1:]:
        s=s+','+str(e)
    f.write(s+'\n')
    f.close()
    
def normalize(nrecords):
    # normalize data
    n=len(nrecords)
    d=len(nrecords[0])
    for j in range(d):
        mx=0.0
        for i in range(n):
            mx=max(mx,abs(nrecords[i][j]))
        for i in range(n):
            nrecords[i][j]/=mx
            
def remove_zero_cols(nrecords):
    # remove zero-features from data
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
    
    
def add_features(v): 
    # add pairwise product features to vector
    w=v[:]
    v2=v[:]
    d=len(v)
        
    for i in range(d):
        for j in range(i+1):
            w.append(v2[i]*v2[j])
    return w    