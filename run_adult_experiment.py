import numpy as np
import numpy.linalg as la
import sys
import prepare_adult_data
import sampling_methods
from misc import *


csv_samples_header=['protected_class','sample_method','sample_size','sample_no','protected_frequency','log_geom_diversity']

csv_table_header=['protected_class','sample_method','sample_size','no_of_samples','protected_freq_mean','entropy_mean','log_diversity_mean','Prob_>_50', 'protected_freq_std','entropy_std','log_diversity_std']

    
    
def add_sample(path,S,record,reg_nrecords,labels):
    l=[labels[i] for i in S]
    fo=fraction_ones(l)
    record.append(fo)
    logd=log_diversity(np.array([reg_nrecords[i] for i in S]))
    record.append(logd)
    write_to_csv(path,record)
    return (fo,logd)
    
    

def run_tests(SAMPLE_SIZE,NO_SAMPLES,PATH_CSV,PATH_CSV_SUMMARY,reg_nrecords,protected_class,labels):
    # run experiments and save results in two CSV files
    # first with info about all samples
    # second with cumulative statistics over all samples
    DATA_SIZE=len(reg_nrecords)
    
    # clean up and prepare headers for the csv files with results
    clean_file(PATH_CSV)
    clean_file(PATH_CSV_SUMMARY)
    write_to_csv(PATH_CSV,csv_samples_header)
    write_to_csv(PATH_CSV_SUMMARY,csv_table_header)
    
    Y=np.array(reg_nrecords)
    M=range(0,DATA_SIZE)
    
    label0=filter(lambda i: labels[i]==0, M)
    label1=filter(lambda i: labels[i]==1, M)
    fraction_protected=fraction_ones(labels)
    prop_sample_size=int(fraction_protected*SAMPLE_SIZE)
    #different sampling methods to be run in this experiment
   
    sample_methods=['everything','uniform','k-uniform','k-uniform-proportional',
        'k-DPP','ki-DPP','ki-DPP-proportional','P-DPP','P-DPP-proportional']
        
        
    for method in sample_methods:
    
        #arrays for gathering statistics over samples
        fos=[]
        logds=[]
        
        print('method='+method)
        for sample_no in range(NO_SAMPLES):
            if method=='everything':
                S=M
            elif method=='uniform':
                S=sampling_methods.uniform_sample(M,SAMPLE_SIZE)
            elif method=='k-uniform':
                S=sampling_methods.uniform_sample(label0,SAMPLE_SIZE/2)+sampling_methods.uniform_sample(label1,SAMPLE_SIZE/2)
            elif method=='k-uniform-proportional':
                S=sampling_methods.uniform_sample(label0,SAMPLE_SIZE-prop_sample_size)+sampling_methods.uniform_sample(label1,prop_sample_size)
            elif method=='k-DPP':
                S=sampling_methods.kDPPGreedySample(Y,SAMPLE_SIZE)
            elif method=='k-DPP-MCMC':
                S=sampling_methods.kDPPSampleMCMC(Y,SAMPLE_SIZE,DATA_SIZE*10)
            elif method=='ki-DPP':
                S=sampling_methods.kiDPPGreedySample(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[labels[e] for e in M])
            elif method=='ki-DPP-proportional':
                S=sampling_methods.kiDPPGreedySample(Y,[SAMPLE_SIZE-prop_sample_size,prop_sample_size],[labels[e] for e in M])
            elif method=='P-DPP':
                S=sampling_methods.PartitionDPPGreedySample(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[labels[e] for e in M])
            elif method=='P-DPP-MCMC':
                S=sampling_methods.PartitionDPPSampleMCMC(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[labels[e] for e in M],DATA_SIZE*10)
            elif method=='P-DPP-proportional':
                S=sampling_methods.PartitionDPPGreedySample(Y,[SAMPLE_SIZE-prop_sample_size,prop_sample_size],[labels[e] for e in M])
            elif method=='old-P-DPP':
                S=sampling_methods.OldPartitionDPPGreedySample(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[labels[e] for e in M])
            else:
                print('error -- method not recognized')
                sys.exit(0)
            (fo,logd)=add_sample(PATH_CSV,S,[protected_class,method,len(S),sample_no],reg_nrecords,labels)
            fos.append(fo)
            logds.append(logd)
            if method=='everything':
                break
                
        # calculate summary of entropy over all samples          
        entropies=np.array([H_entropy(p) for p in fos])
        
        # calculate summary of >50% statistic
        frac_50=0.0
        for p in fos:
            if (abs(p-0.5)<0.001): frac_50+=0.5/len(fos)
            elif p<0.5: frac_50+=0.0
            else: frac_50+=1.0/len(fos)
        
        # write summary to a CSV file
        write_to_csv(PATH_CSV_SUMMARY,
            [protected_class,method,len(S),NO_SAMPLES,np.array(fos).mean(),
            np.array(entropies).mean(),np.array(logds).mean(),frac_50,np.array(fos).std(),
            np.array(entropies).std(),np.array(logds).std()])
    
    

      
def run_adult(prot_class):
    # run the experiment on the Adult data set using prot_class as the protected attribute
    
    DATA_SIZE=5000
    SAMPLE_SIZE=100
    NO_SAMPLES=100
    PATH_CSV='adult-samples'+prot_class+'.csv'
    PATH_CSV_SUMMARY='adult-summary'+prot_class+'.csv'
    
    
    # load the data set from file
    # use https://github.com/mbilalzafar/fair-classification
    nrecords,labels,classes=prepare_adult_data.load_adult_data(DATA_SIZE)
    nrecords=nrecords.tolist()
    nrecords=remove_zero_cols(nrecords)
    normalize(nrecords)
    gender=classes['sex']

    
    # prepare 0-1 arrays determining the gender and the race of the data points
    gender=map(int,gender)
    for i in range(len(gender)):
        gender[i]=1-gender[i]
    race=[]
    for e in classes['race']:
        if (e!=4): race.append(1)
        else: race.append(0)
        
    # enrich the data vectors by adding pairwise product features  
    reg_nrecords=np.array(nrecords).copy().tolist()
    for e in range(len(nrecords)):
        reg_nrecords[e]=add_features(nrecords[e])
    reg_nrecords=remove_zero_cols(reg_nrecords)
    X=np.asarray(nrecords)
    
    
    
    # run experiments
    if (prot_class=='gender-female'):
        run_tests(SAMPLE_SIZE,NO_SAMPLES,PATH_CSV,PATH_CSV_SUMMARY,reg_nrecords,prot_class,gender)
    elif (prot_class=='race-non-white'):
        run_tests(SAMPLE_SIZE,NO_SAMPLES,PATH_CSV,PATH_CSV_SUMMARY,reg_nrecords,prot_class,race)
 


 
run_adult('race-non-white')
run_adult('gender-female')

