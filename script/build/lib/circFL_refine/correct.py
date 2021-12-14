'''
Usage: circFL-refine correct -f circ -g genome -d donor_model -a acceptor_model [-e mistake] [-i dist] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     Evaluation result of full-length circRNA, output of evaluate commond.
    -g genome                   Fasta file of genome.
    -d donor_model              Donor model file, output of train commond.
    -a acceptor_model           Acceptor model file, output of train commond.
    -e mistake=k                Isoforms with less than k mistaken splice site to be corrected.[default: 1].
    -i dist                     Maximum distance of nearby predicted splice site [default: 20].
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

def correct2bed(df):
    new_df=df.iloc[:,0:12].copy()
    new_df_blockNum=[]
    new_df_blockSize=[]
    new_df_blockStart=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        strand=each[5]
        start=each[1]
        end=each[2]
        donorSite=[int(i) for i in each['correct_donorSite'].split(',')] # 1-based
        acceptorSite=[int(i) for i in each['correct_acceptorSite'].split(',')] # 1-based
        if strand=='+':
            left=donorSite
            right=acceptorSite
        else:
            left=acceptorSite
            right=donorSite
        new_blockStart=np.array([start+1]+right)-start-1
        new_blockSize=np.array(left+[end])-np.array([start+1]+right)+1
        new_df_blockNum.append(len(new_blockSize))
        new_blockStart=','.join([str(i) for i in np.array(new_blockStart)])
        new_blockSize=','.join([str(i) for i in np.array(new_blockSize)])
        new_df_blockSize.append(new_blockSize)
        new_df_blockStart.append(new_blockStart)
    new_df.iloc[:,9]=new_df_blockNum
    new_df.iloc[:,10]=new_df_blockSize
    new_df.iloc[:,11]=new_df_blockStart
    new_df.iloc[:,3]=new_df.iloc[:,0]+'|'+new_df.iloc[:,1].map(str)+'|'+new_df.iloc[:,2].map(str)+'|'+new_df.iloc[:,10]+'|'+new_df.iloc[:,11]
    new_df=new_df.drop_duplicates(['name'])
    new_df.index=range(new_df.shape[0])
    return(new_df)

def getPosLR_donor(each,donorSite,acceptorSite,i):
    if each['strand']=='+':
        if i==0:
            posl=each['chromStart']+1
            posr=acceptorSite[i]
        elif i==len(donorSite)-1:
            posl=acceptorSite[i-1]
            posr=acceptorSite[i]
        else:
            posl=acceptorSite[i-1]
            posr=acceptorSite[i]
    else:
        if i==len(donorSite)-1:
            posl=acceptorSite[i]
            posr=each['chromEnd']
        elif i==0:
            posl=acceptorSite[i]
            posr=acceptorSite[i+1]
        else:
            posl=acceptorSite[i]
            posr=acceptorSite[i+1]
    return(posl,posr)

def getPosLR_acceptor(each,donorSite,acceptorSite,i):
    if each['strand']=='-':
        if i==0:
            posl=each['chromStart']+1
            posr=donorSite[i]
        elif i==len(donorSite)-1:
            posl=donorSite[i-1]
            posr=donorSite[i]
        else:
            posl=donorSite[i-1]
            posr=donorSite[i]
    else:
        if i==len(donorSite)-1:
            posl=donorSite[i]
            posr=each['chromEnd']
        elif i==0:
            posl=donorSite[i]
            posr=donorSite[i+1]
        else:
            posl=donorSite[i]
            posr=donorSite[i+1]
    return(posl,posr)

def getCandidate_donor(pos0,posl,posr,can,strand,intronMin=30,intronMax=200000,exonMin=30,exonMax=3000):
    can_list=[]
    for i,value in enumerate(can):
        if value>posl and value<posr:
            if strand=='+':
                exonLen=value-posl+1
                intronLen=posr-value-1
            else:
                exonLen=posr-value+1
                intronLen=value-posl-1
            if exonLen>exonMin and exonLen<exonMax and intronLen>intronMin and intronLen<intronMax:
                can_list.append(value)
    if len(can_list)==0:
        return('')
    elif len(can_list)==1:
        return(can_list[0])
    else:
        minus_can=[]
        for i in can_list:
            minus_can.append(abs(i-pos0))
        return(can_list[sorted(zip(minus_can,range(len(minus_can))))[0][1]])

def getCandidate_acceptor(pos0,posl,posr,can,strand,intronMin=30,intronMax=200000,exonMin=30,exonMax=3000):
    can_list=[]
    for i,value in enumerate(can):
        if value>posl and value<posr:
            if strand=='-':
                exonLen=value-posl+1
                intronLen=posr-value-1
            else:
                exonLen=posr-value+1
                intronLen=value-posl-1
            if exonLen>exonMin and exonLen<exonMax and intronLen>intronMin and intronLen<intronMax:
                can_list.append(value)
    if len(can_list)==0:
        return('')
    elif len(can_list)==1:
        return(can_list[0])
    else:
        minus_can=[]
        for i in can_list:
            minus_can.append(abs(i-pos0))
        return(can_list[sorted(zip(minus_can,range(len(minus_can))))[0][1]])

def correctSplice(each,donor,acceptor):
    donorSite=[int(i) for i in each['donorSite'].split(',')]
    acceptorSite=[int(i) for i in each['acceptorSite'].split(',')]
    donorPredict=[float(i) for i in each['donorPredict'].split(',')]
    acceptorPredict=[float(i) for i in each['acceptorPredict'].split(',')]
    
    donor=[int(i.split('|')[1]) for i in donor]
    acceptor=[int(i.split('|')[1]) for i in acceptor]
    new_donorSite=[]
    new_acceptorSite=[]
    for i,value in enumerate(donorPredict):
        if value<0.5:
            posl,posr=getPosLR_donor(each,donorSite,acceptorSite,i)
            new_value=getCandidate_donor(donorSite[i],posl,posr,donor,each['strand'])
            if new_value!='':
                new_donorSite.append(new_value)
            else:
                return('','')
        else:
            new_donorSite.append(donorSite[i])
    for i,value in enumerate(acceptorPredict):
        if value<0.5:
            posl,posr=getPosLR_acceptor(each,donorSite,acceptorSite,i)
            new_value=getCandidate_acceptor(acceptorSite[i],posl,posr,acceptor,each['strand'])
            if new_value!='':
                new_acceptorSite.append(new_value)
            else:
                return('','')
        else:
            new_acceptorSite.append(acceptorSite[i])
    new_donorSite=','.join([str(i) for i in new_donorSite])
    new_acceptorSite=','.join([str(i) for i in new_acceptorSite])
    return(new_donorSite,new_acceptorSite)

def filterMistake(df,num):
    passIdx=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        donorPredict=sum([float(i)<0.5 for i in each['donorPredict'].split(',')])
        acceptorPredict=sum([float(i)<0.5 for i in each['acceptorPredict'].split(',')])
        if (donorPredict+acceptorPredict)<=num and (donorPredict+acceptorPredict)>0:
            passIdx.append(i)
    new_df=df.iloc[passIdx,:].copy()
    new_df.index=range(new_df.shape[0])
    return(new_df)

def getCorrectDf(df):
    ndf=df.loc[~((df.donorSite==df.correct_donorSite) & (df.acceptorSite==df.correct_acceptorSite)),:].copy()
    return(ndf)
def getDist(df):
    mylist=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        dS=[int(i) for i in each['donorSite'].split(',')] # 1-based
        aS=[int(i) for i in each['acceptorSite'].split(',')] # 1-based
        dS_c=[int(i) for i in each['correct_donorSite'].split(',')] # 1-based
        aS_c=[int(i) for i in each['correct_acceptorSite'].split(',')] # 1-based
        d_diff=[abs(dS[i]-dS_c[i]) for i in range(len(dS))]
        a_diff=[abs(aS[i]-aS_c[i]) for i in range(len(aS))]
        d_max=max(d_diff)
        a_max=max(a_diff)
        mylist.append([sum([i!=0 for i in d_diff]),sum([i!=0 for i in a_diff]),d_max,a_max])
    return(mylist)

def filterDistIdx(df,mis):
    df.index=range(df.shape[0])
    nf=pd.DataFrame(getDist(df),columns=['dN','aN','dM','aM'])
    nf.index=range(nf.shape[0])
    nf=pd.concat([df,nf],axis=1)
    nf_sub=nf.loc[((nf.dN+nf.aN)<=mis)& ((nf.dN+nf.aN)>=1) & ((nf.dM<=20) & (nf.aM<=20)),:].copy()
    return(nf_sub)
def correct(options):
    global genome
    fl_file=options['-f']
    genomeFile=options['-g']
    outDir=options['-o']
    acceptor_file=options['-a']
    donor_file=options['-d']
    distSS=min(200,int(options['-i']))
    misSS=min(5,int(options['-e']))
    
    plog('Check model file')
    fileCheck(acceptor_file)
    fileCheck(donor_file)
    acceptor_model=read_model(acceptor_file)
    donor_model=read_model(donor_file)
    
    plog('Check genome file')
    genome=readFaFile(genomeFile)
    plog('Check circRNA file')
    fl_eval=readEvalFile(fl_file)
    
    plog('Check output file')
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if outDir[-1]!='/':
        outPrefix=outDir+'/'
    else:
        outPrefix=outDir
    correct_outPrefix=outPrefix+'correct/'
    createDir(outPrefix);createDir(correct_outPrefix)
    
    plog('Correct full-length circRNA')
    fl_fail=fl_eval.loc[(fl_eval.donorPredictMin<0.5) | (fl_eval.acceptorPredictMin<0.5),:]
    fl_fail.index=range(fl_fail.shape[0])
    if misSS>0:
        fl_fail=filterMistake(fl_fail,misSS)
    
    circ_df=fl_fail.iloc[:,[0,1,2,5]]
    circ_df.columns=['chr','start','end','strand']

    df=filterDf_new(circ_df)
    df.loc[:,'key']=df.chr+'|'+df.start.map(str)+'|'+df.end.map(str)
    df=df.iloc[:,0:4]
    
    df_sense=df.loc[df.loc[:,'strand']=='+',:].copy()
    df_antisense=df.loc[df.loc[:,'strand']=='-',:].copy()
    
    df_sense_key=(df_sense.chr+'|'+df_sense.start.map(str)+'|'+df_sense.end.map(str)).tolist()
    df_antisense_key=(df_antisense.chr+'|'+df_antisense.start.map(str)+'|'+df_antisense.end.map(str)).tolist()
    
    fl_fail_sense=fl_fail.loc[df_sense.index.tolist(),:].copy()
    fl_fail_antisense=fl_fail.loc[df_antisense.index.tolist(),:].copy()
    
    df_sense.index=range(df_sense.shape[0])
    df_antisense.index=range(df_antisense.shape[0])
    fl_fail_sense.index=range(fl_fail_sense.shape[0])
    fl_fail_antisense.index=range(fl_fail_antisense.shape[0])

    s_donor_pos_list,s_acceptor_pos_list,s_donor_predict_list,s_acceptor_predict_list=extendPredProb_dist(df_sense,'+',genome,donor_model,acceptor_model,fl_fail_sense,distSS)
    a_donor_pos_list,a_acceptor_pos_list,a_donor_predict_list,a_acceptor_predict_list=extendPredProb_dist(df_antisense,'-',genome,donor_model,acceptor_model,fl_fail_antisense,distSS)

    
    
    new_df=[]
    for idx in range(fl_fail_sense.shape[0]):
        tmp_donor=s_donor_pos_list[idx]
        tmp_acceptor=s_acceptor_pos_list[idx]
        each=fl_fail_sense.iloc[idx,:]
        donor_correct,acceptor_correct=correctSplice(each,tmp_donor,tmp_acceptor)
        new_each=each.tolist()
        new_each.extend([donor_correct,acceptor_correct])
        new_df.append(new_each)

    for idx in range(fl_fail_antisense.shape[0]):
        tmp_donor=a_donor_pos_list[idx]
        tmp_acceptor=a_acceptor_pos_list[idx]
        each=fl_fail_antisense.iloc[idx,:]
        donor_correct,acceptor_correct=correctSplice(each,tmp_donor,tmp_acceptor)
        new_each=each.tolist()
        new_each.extend([donor_correct,acceptor_correct])
        new_df.append(new_each)

    new_df=pd.DataFrame(new_df)
    new_df=new_df.loc[new_df.iloc[:,20]!='',:]
    new_df.columns=['chrom','chromStart','chromEnd','name','score','strand','thickStart','thickEnd','itemRgb','blockCount','blockSizes','blockStarts','donorSite','donorMotif','donorPredict','donorPredictMin','acceptorSite','acceptorMotif','acceptorPredict','acceptorPredictMin','correct_donorSite','correct_acceptorSite']
    new_df=getCorrectDf(new_df)
    new_df=filterDistIdx(new_df,misSS)
    new_df.to_csv(correct_outPrefix+'fl_correct.detail',index=None,sep='\t')
    fl_correct_bed=correct2bed(new_df)
    fl_correct_bed.to_csv(correct_outPrefix+'fl_correct.bed',index=None,sep='\t')
    plog('All done!!!')