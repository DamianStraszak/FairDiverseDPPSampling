import math
import random
import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import sys
import time

def VerifyPartitionConstraints(Pvec,kvec,S):
    b=1
    p=len(list(set(Pvec)))
    Svec=[0]*p
    for i in S:
        pVal=Pvec[i]
        Svec[pVal]=Svec[pVal]+1
    for i in range(0,p):
        if(Svec[i]>kvec[i]):
            b=0
    return b

def PartitionDPPMaxGreedy(Y,kvec,Pvec):
    X=Y.copy()
    n=int(X.shape[0])
    p=len(list(set(Pvec)))
    nvec=[0]*p
    cvec=[0]*p
    for i in Pvec:
        nvec[i]=nvec[i]+1
    k=sum(kvec)
    # print(n)
    S=[]
    for i in range(k):
        vals=[0]*n
        for j in range(n):
            if(cvec[Pvec[j]]+1<=kvec[Pvec[j]]):    
                vals[j]=pow(la.norm(X[j,:]),2)
            else:
                vals[j]=0
        ind=np.argmax(vals)
        Xind=X[ind,:].copy()
        normInd=pow(la.norm(X[ind,:]),2)
        S.append(ind)
        for j in range(n):
            X[j,:]=X[j,:] - (np.dot(Xind,np.transpose(X[j,:]))/normInd)*Xind
        cvec[Pvec[ind]]+=1
    return S

def PartitionDPPGreedySample(Y,kvec,Pvec):
    start_time = time.time()
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
        if(multinomSum<0.000000001):
            print('badcase')
            print(len(S))
            print([Pvec[i] for i in S])
            print([Pvec[i] for i in S].count(0))
            print([Pvec[i] for i in S].count(1))
            print(Pvec.count(1))
            sys.exit(0)
            break
        multinom=multinom/multinomSum
        ind=nprand.multinomial(1,multinom)
        ind=np.where(ind==1)
        ind=ind[0][0]
        S.append(ind)
        cvec[Pvec[ind]]+=1
        Xind=X[ind,:].copy()
        normInd=pow(la.norm(X[ind,:]),2)
        #normInd=pow(la.norm(X[ind,:]),1)
        for j in range(n):
            # print('shapes',X[ind,:].shape,'-',X[j,:].shape)
            X[j,:]=X[j,:] - (np.dot(Xind,np.transpose(X[j,:]))/normInd)*Xind
    print("Partition Sampling Running Time--- %s seconds ---" % (time.time() - start_time))
    return S
    
def OldPartitionDPPGreedySample(Y,kvec,Pvec):
    start_time = time.time()
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
                multinom[j]=pow(la.norm(X[j,:]),2)
            else:
                multinom[j]=0
        multinomSum=sum(multinom)
        if(multinomSum<0.000000001):
            print('badcase')
            print(len(S))
            print([Pvec[i] for i in S])
            print([Pvec[i] for i in S].count(0))
            print([Pvec[i] for i in S].count(1))
            print(Pvec.count(1))
            sys.exit(0)
            break
        multinom=multinom/multinomSum
        ind=nprand.multinomial(1,multinom)
        ind=np.where(ind==1)
        ind=ind[0][0]
        S.append(ind)
        cvec[Pvec[ind]]+=1
        Xind=X[ind,:].copy()
        normInd=pow(la.norm(X[ind,:]),2)
        #normInd=pow(la.norm(X[ind,:]),1)
        for j in range(n):
            # print('shapes',X[ind,:].shape,'-',X[j,:].shape)
            X[j,:]=X[j,:] - (np.dot(Xind,np.transpose(X[j,:]))/normInd)*Xind
    print("Partition Sampling Running Time--- %s seconds ---" % (time.time() - start_time))
    return S

def PartitionDPPSampleMCMC(Y,kvec,Pvec,numIter=10):
    start_time = time.time()
    X=Y.copy()
    K= np.dot(X,np.transpose(X))
    P0=np.where(Pvec==0)
    P0=P0[0]
    P1=np.where(Pvec==1)
    P1=P1[0]
    S=[]
    # S=PartitionDPPMaxGreedy(X,kvec,Pvec)
    S=PartitionDPPGreedySample(X,kvec,Pvec)
    m = K.shape[0]
    k=sum(kvec)
    Spr=set(S)
    Sbar=list(set(range(0,m))-Spr)
    t=0
    while t<numIter:
        # print('t',t)
        outIndex=random.randrange(0,k)
        outElt=S[outIndex]

        inIndex=random.randrange(0,m-k)
        inElt=Sbar[inIndex]
        
        if(Pvec[Sbar[inIndex]]==Pvec[outElt]):
            t+=1
            if np.random.ranf()<0.5: continue
            T=list(S)
            T[outIndex]=inElt
            p=min(1,la.det(K[T,:][:,T])/la.det(K[S,:][:,S]))
            p=max(0,p)
            outcome=nprand.binomial(1,p,1)
            if(outcome[0]==1):
                S[outIndex]=inElt
                Sbar[inIndex]=outElt
    print("PartiotionDPP Sampling Running Time--- %s seconds ---" % (time.time() - start_time))
    return S




