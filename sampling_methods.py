import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import random

def uniform_sample(elems,k):
    n=len(elems)
    l=range(0,n)
    random.shuffle(l)
    return [elems[i] for i in l[:k]]
    
    
def kDPPGreedySample(Y,k):
    X=Y.copy()
    n=int(X.shape[0])
    S=[]
    while len(S)<k:
        multinom=[0]*n
        for j in range(n):
            multinom[j]=pow(la.norm(X[j,:]),2)
        multinomSum=sum(multinom)
        if(multinomSum<1e-9):
            raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
        multinom=multinom/multinomSum
        ind=nprand.multinomial(1,multinom)
        ind=np.where(ind==1)
        ind=ind[0][0]
        S.append(ind)
        Xind=X[ind,:].copy()
        normInd=pow(la.norm(X[ind,:]),2)
        for j in range(n):
            X[j,:]=X[j,:] - (np.dot(Xind,np.transpose(X[j,:]))/normInd)*Xind
    return S
      
def PartitionDPPGreedySample(Y,kvec,Pvec):
    X=Y.copy()
    n=int(X.shape[0])
    S=[]
    k=sum(kvec)
    p=len(kvec)
    cvec=[0]*p
    for i in range(k):
        multinom=[0]*n
        for j in range(n):
            if(cvec[Pvec[j]]+1<=kvec[Pvec[j]]):
                multinom[j]=pow(la.norm(X[j,:]),2)*((kvec[Pvec[j]]-cvec[Pvec[j]])*1.0/kvec[Pvec[j]])
            else:
                multinom[j]=0
        multinomSum=sum(multinom)
        if(multinomSum<1e-9):
            raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
        multinom=multinom/multinomSum
        ind=nprand.multinomial(1,multinom)
        ind=np.where(ind==1)
        ind=ind[0][0]
        S.append(ind)
        cvec[Pvec[ind]]+=1
        Xind=X[ind,:].copy()
        normInd=pow(la.norm(X[ind,:]),2)
        for j in range(n):
            X[j,:]=X[j,:] - (np.dot(Xind,np.transpose(X[j,:]))/normInd)*Xind
    return S
    
def kiDPPGreedySample(Y,kvec,Pvec):
    X=Y.copy()
    n=int(X.shape[0])
    p=len(kvec)
    P=[[] for e in kvec]
    for i in range(n):
        P[Pvec[i]].append(i)
    S=[]
    for i in range(p):
        M=P[i]
        X0=X[M,:].copy()    
        S0=kDPPGreedySample(X0,kvec[i])
        S=S+[M[e] for e in S0]
    ksum=0
    for ki in kvec:
        ksum+=ki
    if len(S)!=ksum:
        raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
    return S
    
    

    
