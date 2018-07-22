# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.cluster import KMeans




def write_vectors_to_file(vectors,female,scientist):
    # memoize the result in a file to not recompute it every time
    N=len(vectors)
    path='image-vectors\\image_data.txt'
    f=open(path,'w')
    f.close()
    f=open(path,'a')
    for i in range(N):
        s=str(vectors[i][0])
        for j in range(1,len(vectors[i])):
            s=s+','+str(vectors[i][j])
        s=s+','+str(female[i])+','+str(scientist[i])
        f.write(s+'\n')
    f.close()
    
    
def read_vectors_from_file():
    # the file 'image-vectors\image_data.txt' constains image vectors in the following format
    # every row contains one vector, the last two entries determine gender and occupation
    path='image-vectors\\image_data.txt'
    vectors=[]
    female=[]
    scientist=[]
    f=open(path)
    while True:
        l=f.readline()
        if l=='': break
        s=map(float,l.split(','))
        n=len(s)
        vectors.append(s[:n-2])
        female.append(int(s[n-2]))
        scientist.append(int(s[n-1]))
    f.close()
    return (vectors,female,scientist)
    

def read_dataset_sift():  
    # use sift descriptors of the images (in image-vectors\...)
    # to construct feature vectors for these images using k-means (as described in the paper)
    data=[]
    female=[]
    scientist=[]
    if os.path.exists('.\image-vectors\image_data.txt'):
        return read_vectors_from_file()
    for filename in os.listdir('.\images\PainterFemale'):
        if filename.endswith(".txt"):
            fpath=os.path.join('.\images\PainterFemale', filename)
            f=open(fpath)
            bag=[]
            while True:
                l=f.readline()
                if l=='': break
                s=map(float,l.split(','))
                bag.append(s)
            scientist.append(0)
            female.append(1)
            data.append(bag)
               
    for filename in os.listdir('.\images\PainterMale'):
        if filename.endswith(".txt"):
            fpath=os.path.join('.\images\PainterMale', filename)
            f=open(fpath)
            bag=[]
            while True:
                l=f.readline()
                if l=='': break
                s=map(float,l.split(','))
                bag.append(s)
            scientist.append(0)
            female.append(0)
            data.append(bag)
                
    for filename in os.listdir('.\images\ScientistFemale'):
        if filename.endswith(".txt"):
            fpath=os.path.join('.\images\ScientistFemale', filename)
            f=open(fpath)
            bag=[]
            while True:
                l=f.readline()
                if l=='': break
                s=map(float,l.split(','))
                bag.append(s)
            scientist.append(1)
            female.append(1)
            data.append(bag)
                
    for filename in os.listdir('.\images\ScientistMale'):
        if filename.endswith(".txt"):
            fpath=os.path.join('.\images\ScientistMale', filename)
            f=open(fpath)
            bag=[]
            while True:
                l=f.readline()
                if l=='': break
                s=map(float,l.split(','))
                bag.append(s)
            scientist.append(1)
            female.append(0)
            data.append(bag)
             
    X=[]
    for Y in data:
        for e in Y:
            # subsampling to avoid large number of vectors for k-means
            if np.random.randint(0,10)==0:
                X.append(e)
    X=np.array(X)
    
    kmeans = KMeans(n_clusters=128, random_state=0).fit(X)      
    C=kmeans.cluster_centers_
    print('kmeans done')
    records=[]
    print(X.shape)
    cnt=0
    for Y in data:
        x=np.array([0.0]*len(C))
        if (cnt%10==0): print(cnt,' out of ',len(data))
        cnt+=1
        d=len(Y)
        for z in Y:
            best=0
            distance=1e10
            for i in range(len(C)):
                if (np.linalg.norm(z-C[i])<distance):
                    best=i
                    distance=np.linalg.norm(z-C[i]) 
            x[best]+=1.0
        x=x/np.linalg.norm(x)
        records.append(x.tolist())
    write_vectors_to_file(records,female,scientist)
    return (records,female,scientist)
            
    
def read_dataset():  
    # outputs the image data in a convenient form
    data=[]
    female=[]
    scientist=[]
    
    file_name="image-vectors\\PainterFemale.txt"
    f=open(file_name)
    while True:
        l=f.readline()
        if l=='': break
        s=map(float,l.split(','))
        data.append(s)
        scientist.append(0)
        female.append(1)
    print(len(data))
    
    file_name="image-vectors\\PainterMale.txt"
    f=open(file_name)
    while True:
        l=f.readline()
        if l=='': break
        s=map(float,l.split(','))
        data.append(s)
        scientist.append(0)
        female.append(0)
    print(len(data))
    
    file_name="image-vectors\\ScientistFemale.txt"
    f=open(file_name)
    while True:
        l=f.readline()
        if l=='': break
        s=map(float,l.split(','))
        data.append(s)
        scientist.append(1)
        female.append(1)
    print(len(data))
    
    file_name="image-vectors\\ScientistMale.txt"
    f=open(file_name)
    while True:
        l=f.readline()
        if l=='': break
        s=map(float,l.split(','))
        data.append(s)
        scientist.append(1)
        female.append(0)
    print(len(data))
    
    return (data,female,scientist)
    
    
    
  

 

 

















