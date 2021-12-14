'''
Usage: circFL-refine construct -f circ -g genome [-d donor_model] [-a acceptor_model] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     File of circRNA BSJ position in tab format (the first four columns are chr, 0-based start, 1-based end, optional strand).
    -g genome                   Fasta file of genome.
    -d donor_model              Donor model file, output of train commond.
    -a acceptor_model           Acceptor model file, output of train commond.
    -o output                   Output dir [default: circFL_refine].
'''

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

def identifyStrand(df):
    strand=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        chr=each['chr']
        start=each['start']
        end=each['end']
        allChr=list(genome.keys())
        if chr in allChr:
            left=genome.sequence({'chr': chr, 'start':start-2, 'stop':start-1,}).upper()
            right=genome.sequence({'chr': chr, 'start':end+1, 'stop':end+2,}).upper()
            if left=='AG' and right=='GT':
                strand.append('+')
            elif left=='AC' and right=='CT':
                strand.append('-')
            else:
                strand.append('.')
        else:
            strand.append('.')
    return(strand)
    



def aroundSeq(pos,strand,genome,around=75):
    detail=pos.split('|')
    chr=detail[0]
    pos=int(detail[1])
    seq=genome.sequence({'chr': chr, 'start':pos-around, 'stop':pos+around,'strand':strand}).upper()
    return(seq)


    
def allcomb(start,end,leftSite,rightSite,intronMin=30,intronMax=200000,exonMin=30,exonMax=3000):
    eiMin=intronMin+exonMin
    eiMax=intronMax+exonMax 
    list0=[]
    for i,l in enumerate(leftSite):
        list1=[]
        each_exonLen=l-start        
        if (each_exonLen>exonMin) and (each_exonLen<exonMax):
            for j,r in enumerate(rightSite):
                list2=[]
                each_intronLen=r-l
                if (each_intronLen>intronMin) and (each_intronLen<intronMax):
                    last_exonLen=end-r
                    if i==(len(leftSite)-1):
                        if last_exonLen<exonMax and last_exonLen>exonMin:
                            list2.append([[l,r]])
                        else:
                             list2.append([[-1,-1]])
                    elif j==(len(rightSite)-1):
                        if last_exonLen<exonMax and last_exonLen>exonMin:
                            list2.append([[l,r]])
                        else:
                             list2.append([[-1,-1]])
                    else:
                        adj_leftSite=leftSite[(i+1):]
                        adj_rightSite=rightSite[(j+1):]
                        adj_start=r
                        adj_end=end    
                        subcom=allcomb(adj_start,adj_end,adj_leftSite,adj_rightSite)
                        if last_exonLen>exonMin:
                            for p in subcom:
                                if len(p)==0:
                                    list2.append([[l,r]]+[[]])
                                else:
                                    for m in p:
                                        if len(m)==0:
                                            list2.append([[l,r]]+[[]])
                                        else:
                                            for n in m:
                                                if len(n)==0:
                                                    list2.append([[l,r]]+[[]])
                                                else:
                                                    list2.append([[l,r]]+n)
                        else:
                            list2.append([[-1,-1]])
                list1.append(list2)
        list0.append(list1)
    #print(list0)
    return(list0)

def explainSite(arr):
    newarr=[]
    for i in arr:
        if len(i)!=0:
            for j in i:
                if len(j)!=0:
                    for p in j:
                        if len(p)!=0:
                            list1=[]
                            for m in p:
                                if len(m)!=0:
                                    list1.append(m)
                            newarr.append(list1)
    mem_dict={}
    nn=[]
    for i in newarr:
        tmp=[]
        for j in i:
            tmp.extend(str(j))
        tmpKey='|'.join(tmp)
        if not mem_dict.__contains__(tmpKey):
            nn.append(i)
            mem_dict[tmpKey]=1
    return(nn)

def checkSplice(arr,start,end,intronMin=30,intronMax=200000,exonMin=30,exonMax=1000,maxLen=2500,exonNumMax=10):
    newarr=[]
    for i in arr:
        ispass=True
        tmp_exonStart=[start]+[j[1]-1 for j in i]
        tmp_exonEnd=[j[0] for j in i]+[end]
        tmp_exonLen=[]
        for j in range(len(tmp_exonStart)):
            tmp_exonLen.append(tmp_exonEnd[j]-tmp_exonStart[j])
        tmp_intronLen=[j[1]-j[0]-1 for j in i]
        for j in tmp_intronLen:
            if j<intronMin and j>intronMax:
                ispass=False
        for j in tmp_exonLen:
            if j<exonMin and j>exonMax:
                ispass=False
        total_len=sum(tmp_exonLen)
        #print(total_len)
        if total_len>exonMax:
            ispass=False
        if len(tmp_exonLen)>exonNumMax:
            ispass=False
        if ispass:
            newarr.append(i)
    return(newarr)

def formatSplice2bed(chr,start,end,arr,strand):
    newarr=[]
    for i in arr:
        tmp_exonStart=[start]+[j[1]-1 for j in i]
        tmp_exonEnd=[j[0] for j in i]+[end]
        tmp_exonLen=[]
        for j in range(len(tmp_exonStart)):
            tmp_exonLen.append(tmp_exonEnd[j]-tmp_exonStart[j])
        isoID=chr+'|'+','.join([str(j) for j in tmp_exonStart])+'|'+','.join([str(j) for j in tmp_exonEnd])
        tmp_exonStart=[j-start for j in tmp_exonStart]
        exonNum=len(tmp_exonStart)
        newarr.append([chr,str(start),str(end),isoID,'0',strand,'0','0','0',str(exonNum),','.join([str(j) for j in tmp_exonLen]),','.join([str(j) for j in tmp_exonStart])])
    return(newarr)

def selectTop(site,pre,num=9):
    return(sorted([i[1] for i in sorted(zip(pre,site),reverse = True )[0:num]]))

def outFL(df,fout,strand,donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list):
    for i in range(df.shape[0]):
        #if i % 10000==0:
         #   print(i)
        each=df.iloc[i,:]
        start=each['start'] # 0-base
        end=each['end'] # 1-base
        chr=each['chr']
        strand=each['strand']
        if strand=='+':
            leftSite=[int(i.split('|')[1]) for i in donor_pos_list[i]]
            rightSite=[int(i.split('|')[1]) for i in acceptor_pos_list[i]]
        else:
            leftSite=[int(i.split('|')[1]) for i in acceptor_pos_list[i]]
            rightSite=[int(i.split('|')[1]) for i in donor_pos_list[i]]
        leftSite=selectTop(leftSite,donor_predict_list[i])
        rightSite=selectTop(rightSite,acceptor_predict_list[i])
        #print('leftSite:{} rightSite:{}'.format(len(leftSite),len(rightSite)))
        tmp=allcomb(start,end,leftSite,rightSite)
        tmp1=explainSite(tmp)
        if len(tmp1)>0:
            tmp2=checkSplice(tmp1,start,end)
            if len(tmp2)>0:
                tmp3=formatSplice2bed(chr,start,end,tmp2,strand)
                if len(tmp3)>0:
                    for i in tmp3:
                        fout.write('\t'.join(i)+'\n')

def construct(options):
    global genome,acceptor_model,donor_model
    circ_file=options['-f']
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
    circ_df=readPosFile(circ_file).iloc[:,0:4]
    plog('Check output file')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if outDir[-1]!='/':
        outPrefix=outDir+'/'
    else:
        outPrefix=outDir
    predict_outPrefix=outPrefix+'construct/'
    createDir(outPrefix);createDir(predict_outPrefix)
    fileName=predict_outPrefix+'construct_fl.bed'
    fout=open(fileName,'w')
    if circ_df.shape[1]==4:
        circ_df.columns=['chr','start','end','strand']
    else:
        circ_df.columns=['chr','start','end']
        strand=identifyStrand(circ_df)
        circ_df.loc[:,'strand']=strand
    plog('Construct full-length circRNA')
    df=filterDf(circ_df)
    df_sense=df.loc[df.loc[:,'strand']=='+',:].copy()
    df_antisense=df.loc[df.loc[:,'strand']=='-',:].copy()
    donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list=extendPredProb(df_sense,'+',genome,donor_model,acceptor_model)
    outFL(df_sense,fout,'+',donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list)
    donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list=extendPredProb(df_antisense,'-',genome,donor_model,acceptor_model)
    outFL(df_antisense,fout,'-',donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list)
    fout.close()
    plog('All done!!!')
