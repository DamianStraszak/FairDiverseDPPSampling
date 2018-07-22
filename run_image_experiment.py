import numpy as np
import numpy.linalg as la
import read_images
import sys
import sampling_methods
from misc import *


    
def data_sample(path,S,record,reg_nrecords,labels):
    l=[labels[i] for i in S]
    fo=fraction_ones(l)
    record.append(fo)
    logd=log_diversity(np.array([reg_nrecords[i] for i in S]))
    record.append(logd)
    return (fo,logd)
    


def run_images_exp():
    # read image dataset
    (records,female,scientist)=read_images.read_dataset_sift()
    DATA_SIZE=len(records)
    NO_SAMPLES=100
    
    #paths to files where experiment summary is written
    PATH_CSV='images-exp3-samples.csv'
    PATH_CSV_SUMMARY='images-exp3-summary.csv'
    
    clean_file(PATH_CSV)
    clean_file(PATH_CSV_SUMMARY)

    #construct headers for both csv files
    write_to_csv(PATH_CSV,
        ['sample_method','sample_size','sample_no','percent_male',
        'protected_frequency','log_geom_diversity','4_entropy'])
        
    write_to_csv(PATH_CSV_SUMMARY,
        ['sample_method','sample_size','no_of_samples','percent_male','protected_freq_mean',
        'log_diversity_mean','2entropy_mean','4entropy_mean','D_mean','Prob_>_50','protected_freq_std', 'log_diversity_std','2entropy_std','4entropy-std','D_std','Prob_>_50_std'])
        
    
    
    sample_methods=['uniform','k-DPP','ki-DPP','P-DPP']
    SAMPLE_SIZE=40
    for method in sample_methods:
        for PERCENT_MALE in [10,20,30,40,50]:
            M=[]
            male_scientist=PERCENT_MALE*2
            male_artist=PERCENT_MALE*2
            female_scientist=(100-PERCENT_MALE)*2
            female_artist=(100-PERCENT_MALE)*2
            for e in range(0,DATA_SIZE):
                if female[e] and scientist[e] and female_scientist>0:
                    female_scientist-=1
                    M.append(e)
                if female[e] and not scientist[e] and female_artist>0:
                    female_artist-=1
                    M.append(e)
                if not female[e] and scientist[e] and male_scientist>0:
                    male_scientist-=1
                    M.append(e)
                if not female[e] and not scientist[e] and male_artist>0:
                    male_artist-=1
                    M.append(e)
            Y=np.array([records[e] for e in M])
            label0=filter(lambda i: female[i]==0, M)
            label1=filter(lambda i: female[i]==1, M)
            fos=[]
            logds=[]
            entr4=[]
            D4=[]
            print('method='+method)
            for sample_no in range(NO_SAMPLES):
                
                # sample accoriding to a given sampling method
                if method=='everything':
                    S=M
                elif method=='uniform':
                    S=sampling_methods.uniform_sample(M,SAMPLE_SIZE)
                elif method=='k-uniform':
                    S=sampling_methods.uniform_sample(label0,SAMPLE_SIZE/2)+uniform_sample(label1,SAMPLE_SIZE/2)
                elif method=='k-DPP':
                    S=sampling_methods.kDPPGreedySample(Y,SAMPLE_SIZE)
                    S=[M[i] for i in S]
                elif method=='ki-DPP':
                    S=sampling_methods.kiDPPGreedySample(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[female[e] for e in M])
                    S=[M[i] for i in S]
                elif method=='P-DPP':
                    S=sampling_methods.PartitionDPPGreedySample(Y,[SAMPLE_SIZE/2,SAMPLE_SIZE/2],[female[e] for e in M])
                    S=[M[i] for i in S]
                else:
                    print('error -- method not recognized')
                    sys.exit(0)
                
                #construct a record with info about this sample
                record=[method,len(S),sample_no,PERCENT_MALE]
                (fo,logd)=data_sample(PATH_CSV,S,[method,len(S),sample_no,PERCENT_MALE],records,female)
                record.append(fo)
                record.append(logd)
                fos.append(fo)
                logds.append(logd)
                entr4.append(entropy_prob(comp_fractions(S,female,scientist)))
                record.append(entropy_prob(comp_fractions(S,female,scientist)))
                D4.append(KL_dist_to_uniform(comp_fractions(S,female,scientist)))
                
                # write info about this sample to a CSV file
                write_to_csv(PATH_CSV,record)
                if method=='everything':
                    break
                    
                    
            # calculate summary of entropy over all samples        
            entropies=np.array([H_entropy(p) for p in fos])
            
            # calculate summary of >50% statistic 
            frac_50=[]
            for p in fos:
                if (abs(p-0.5)<0.001): 
                    frac_50.append(0.5)
                elif p<0.5: 
                    frac_50.append(0.0)
                else: 
                    frac_50.append(1.0)
            
            # write summary to a CSV file
            write_to_csv(PATH_CSV_SUMMARY,
                [method,len(S),NO_SAMPLES,PERCENT_MALE,np.array(fos).mean(),
                np.array(logds).mean(),np.array(entropies).mean(),np.array(entr4).mean(),
                np.array(D4).mean(),np.array(frac_50).mean(),np.array(fos).std(),
                np.array(logds).std(),np.array(entropies).std(),np.array(entr4).std(),
                np.array(D4).std(),np.array(frac_50).std()])  
    
    

 

run_images_exp()

