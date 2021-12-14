'''
Usage: circFL-refine evaluate -f circ -g genome [-d donor_model] [-a acceptor_model] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     BED format file of full-length circRNA.
    -g genome                   Fasta file of genome.
    -d donor_model              Donor model file, output of train commond.
    -a acceptor_model           Acceptor model file, output of train commond.
    -o output                   Output dir [default: circFL_refine].
'''

# circFL-refine.py evaluate  -f fl_pass.bed -a circFL_refine/train/acceptor_model.net -d circFL_refine/train/donor_model.net -g /media/data4/lzl/genome/hg19/hg19.fa
from .genericFun import *
import sys,time
import re
import argparse
import pandas as pd
import numpy as np
import os
import math
import pyfasta
import torch
import torch.nn as nn

def getflsite(df):
    dict_donor={}
    dict_acceptor={}
    for i in range(df.shape[0]):
        donor=[]
        acceptor=[]
        each=df.iloc[i,:]
        chr=each[0]
        start=each[1]
        end=each[2]
        eLen=[int(i) for i in each[10].split(',')]
        eStart=[int(i) for i in each[11].split(',')]
        strand=each[5]
        if len(eLen)>1:
            for j in range(len(eLen)-1): # strand '+'  donor; '-' acceptor
                subStart=start+eStart[j]
                subEnd=str(subStart+eLen[j])
                if strand=='+':
                    donor.append(chr+'|'+subEnd)
                else:
                    acceptor.append(chr+'|'+subEnd)   
            for j in range(1,len(eLen)):# strand '+'  acceptor; '-' donor
                subStart=str(start+eStart[j]+1)
                if strand=='+':
                    acceptor.append(chr+'|'+subStart)
                else:
                    donor.append(chr+'|'+subStart)
        dict_donor[i]=donor
        dict_acceptor[i]=acceptor
    return(dict_donor,dict_acceptor)

def getMotif(pos,strand,astype):
    pos=pos.split('|')
    chr=pos[0]
    start=int(pos[1])
    if astype =='donor' and strand=='+':
        motif=genome.sequence({'chr': chr, 'start':start+1, 'stop':start+2,'strand':strand}).upper()
    elif astype =='acceptor' and strand=='+':
        motif=genome.sequence({'chr': chr, 'start':start-2, 'stop':start-1,'strand':strand}).upper()
    elif astype =='donor' and strand=='-':
        motif=genome.sequence({'chr': chr, 'start':start-2, 'stop':start-1,'strand':strand}).upper()
    else:
        motif=genome.sequence({'chr': chr, 'start':start+1, 'stop':start+2,'strand':strand}).upper()
    return(motif)

def getModelResult(model,x):
    # evaluation
    y=model(x[0:(0+10000),]).data.cpu().numpy()
    for i in range(10000,x.shape[0],10000):
        y =np.append(y,model(x[i:(i+10000),]).data.cpu().numpy(),axis=0)
    return(list(y[:,0]))

def formatAS(motif,predit,pos_list,pos_dict):
    pos2predit=dict(zip(pos_list,predit))
    out=[]
    for i,value in pos_dict.items():
        tmp=[]
        if len(value)>0:
            tmp_motif=motif[i]
            tmp_predict=[]
            tmp_pos=[]
            for j in value:
                if pos2predit.__contains__(j):
                    tmp_predict.append(pos2predit[j])
                else:
                    tmp_predict.append(-1)
                tmp_pos.append(j.split('|')[1])
            minP=min(tmp_predict)
            tmp_pos=','.join([str(m) for m in tmp_pos])
            tmp_motif=','.join([str(m) for m in tmp_motif])
            tmp_predict=','.join([str(m) for m in tmp_predict])
            
            tmp=[tmp_pos,tmp_motif,tmp_predict,minP]
        out.append(tmp)
    return(out)

def evaluate(options):
    global genome
    fl_file=options['-f']
    genomeFile=options['-g']
    outDir=options['-o']
    acceptor_file=options['-a']
    donor_file=options['-d']
    plog('Check model file')
    fileCheck(acceptor_file)
    fileCheck(donor_file)
    acceptor_model=read_model(acceptor_file)
    donor_model=read_model(donor_file)
    
    plog('Check genome file')
    genome=readFaFile(genomeFile)
    plog('Check circRNA file')
    fl_pass=readBedFile(fl_file)
    
    plog('Check output file')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if outDir[-1]!='/':
        outPrefix=outDir+'/'
    else:
        outPrefix=outDir
    eval_outPrefix=outPrefix+'evaluate/'
    createDir(outPrefix);createDir(eval_outPrefix)
    
    plog('Evaluate full-length circRNA')
    pass_index=fl_pass.iloc[:,0].isin(list(genome.keys())) & fl_pass.iloc[:,5].isin(['+','-'])
    
    fl_pass2=fl_pass.loc[pass_index,:].copy()
    dict_d,dict_a=getflsite(fl_pass2)

    dict_d_motif={};dict_a_motif={}

    for i in range(fl_pass2.shape[0]):
        strand=fl_pass2.iloc[i,:][5]
        tmp=dict_d[i]
        tmp_m=[]
        for j in tmp:
            tmp_m.append(getMotif(j,strand,'donor'))
        dict_d_motif[i]=tmp_m

    for i in range(fl_pass2.shape[0]):
        strand=fl_pass2.iloc[i,:][5]
        tmp=dict_a[i]
        tmp_m=[]
        for j in tmp:
            tmp_m.append(getMotif(j,strand,'acceptor'))
        dict_a_motif[i]=tmp_m
        
    each_seq_donor,each_seq_acceptor,each_seq_donor_pos,each_seq_acceptor_pos=getFLaroundSeq(fl_pass2,genome)

    each_seq_donor_onehot=encodeSeqs(each_seq_donor).astype(np.float32)
    each_seq_acceptor_onehot=encodeSeqs(each_seq_acceptor).astype(np.float32)
    each_seq_donor_torch = torch.from_numpy(each_seq_donor_onehot)
    each_seq_acceptor_torch = torch.from_numpy(each_seq_acceptor_onehot)

    each_donor_predict=getModelResult(donor_model,each_seq_donor_torch)
    each_acceptor_predict=getModelResult(acceptor_model,each_seq_acceptor_torch)
    
    out_donor=formatAS(dict_d_motif,each_donor_predict,each_seq_donor_pos,dict_d)
    out_acceptor=formatAS(dict_a_motif,each_acceptor_predict,each_seq_acceptor_pos,dict_a)
    tmp=[len(out_donor[i])>0 and len(out_acceptor[i])>0 for i in range(len(out_donor))]
    fl_pass3=fl_pass2.loc[tmp,:].copy()
    donor_df=pd.DataFrame(np.array(list(np.array(out_donor)[tmp])))
    acceptor_df=pd.DataFrame(np.array(list(np.array(out_acceptor)[tmp])))
    as_df=pd.DataFrame(pd.concat([donor_df,acceptor_df],axis=1))
    as_df.columns=['donorSite','donorMotif','donorPredict','donorPredictMin','acceptorSite','acceptorMotif','acceptorPredict','acceptorPredictMin']
    fl_pass3.index=range(fl_pass3.shape[0])
    fl_pass4=pd.concat([fl_pass3.iloc[:,0:12],as_df],axis=1)
    fl_pass4.columns=['chrom','chromStart','chromEnd','name','score','strand','thickStart','thickEnd','itemRgb','blockCount','blockSizes','blockStarts','donorSite','donorMotif','donorPredict','donorPredictMin','acceptorSite','acceptorMotif','acceptorPredict','acceptorPredictMin']
    fl_pass4.to_csv(eval_outPrefix+'fl_evaluate.txt',index=None,sep='\t')
    plog('All done!!!')