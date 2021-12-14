import os,sys,time,pyfasta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

BASE_DICT = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

def read_model(fileName):
    bn_state_dict = torch.load(fileName)
    new_bn = Splicenet(151,1).to("cpu")
    new_bn.load_state_dict(bn_state_dict)
    return(new_bn)
    
def plog(x):
    t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('#### %s #### \n%s'   % (t,x))
    sys.stdout.flush()
 
def fileCheck(file):
    if not os.path.exists(file):
        sys.exit('ERROR: %s is not exist!!!' % file)

def readFaFile(fileName):
    if not os.path.exists(fileName):
        sys.exit('ERROR: %s is not exist!!!' % fileName)
    fff=open(fileName)
    fl=fff.readline()
    fff.close()
    if fl[0]!='>':
        sys.exit('ERROR:  %s need to be Fasta format!!!' % fileName)
    try:
        faFile=pyfasta.Fasta(fileName)
        return(faFile)
    except:
        sys.exit('ERROR: make sure %s is fai indexed!!!' % fileName)
def createDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def readBedFile(fileName):
    fin=open(fileName)
    fin_1=fin.readline().split('\t')
    if len(fin_1)<12:
        sys.exit('ERROR:  %s need at least 12 columns!!!' % fileName)
    num_1_9=[str(i) for i in range(1,10)]
    if (fin_1[1][0] not in num_1_9) or (fin_1[2][0] not in num_1_9):
        fl=pd.read_csv(fileName,sep='\t')            
    else:
        fl=pd.read_csv(fileName,sep='\t',header=None)
    fl.columns=range(fl.shape[1])
    return(fl)

def readEvalFile(fileName):
    fin=open(fileName)
    fin_1=fin.readline().split('\t')
    fin.close()
    if len(fin_1)<20:
        sys.exit('ERROR:  %s need at least 20 columns!!!' % fileName)
    num_1_9=[str(i) for i in range(1,10)]
    if (fin_1[1][0] not in num_1_9) or (fin_1[2][0] not in num_1_9):
        fl=pd.read_csv(fileName,sep='\t')            
    else:
        sys.exit('ERROR: make sure %s have header!!!' % fileName)
    return(fl)


def readPosFile(fileName):
    fin=open(fileName)
    fin_1=fin.readline().split('\t')
    if len(fin_1)<3:
        sys.exit('ERROR:  %s need at least 3 columns!!!' % fileName)
    num_1_9=[str(i) for i in range(1,10)]
    if (fin_1[1][0] not in num_1_9) or (fin_1[2][0] not in num_1_9):
        fl=pd.read_csv(fileName,sep='\t')            
    else:
        fl=pd.read_csv(fileName,sep='\t',header=None)
    fl.columns=range(fl.shape[1])
    return(fl)

class Splicenet(nn.Module):
    def __init__(self, sequence_length, n_targets):
        super(Splicenet, self).__init__()
        conv_kernel_size = 6
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                    kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1),
            
            nn.Conv1d(128, 128, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                    kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2))

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size))
        self.classifier = nn.Sequential(
            nn.Linear(128 * self._n_channels, n_targets),
            nn.Sigmoid())

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), -1)
        predict = self.classifier(reshape_out)
        return(predict)
        
def encodeSeqs(seqs):
    inputsize=len(seqs[0])
    seq_encode = np.zeros((len(seqs), 4, inputsize), np.bool_)
    n = 0
    for line in seqs:
        for i, c in enumerate(line[:inputsize]):
            seq_encode[n, :, i] = BASE_DICT[c]
        n = n + 1
    return(seq_encode)
    
def aroundSeq(pos,strand,genome,around=75):
    detail=pos.split('|')
    chr=detail[0]
    pos=int(detail[1])
    seq=genome.sequence({'chr': chr, 'start':pos-around, 'stop':pos+around,'strand':strand}).upper()
    return(seq)
    
def flsite(df):
    donor_plus=[]
    donor_minus=[]
    acceptor_plus=[]
    acceptor_minus=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        chr=each[0]
        start=each[1]
        end=each[2]
        eLen=[int(i) for i in each[10].split(',') if i!='']
        eStart=[int(i) for i in each[11].split(',') if i!='']
        strand=each[5]
        if len(eLen)>1:
            for j in range(len(eLen)-1): # strand '+'  donor; '-' acceptor
                subStart=start+eStart[j]
                subEnd=str(subStart+eLen[j])
                if strand=='+':
                    donor_plus.append(chr+'|'+subEnd)
                else:
                    acceptor_minus.append(chr+'|'+subEnd)   
            for j in range(1,len(eLen)):# strand '+'  acceptor; '-' donor
                subStart=str(start+eStart[j]+1)
                if strand=='+':
                    acceptor_plus.append(chr+'|'+subStart)
                else:
                    donor_minus.append(chr+'|'+subStart)
    return(list(set(donor_plus)),list(set(donor_minus)),list(set(acceptor_plus)),list(set(acceptor_minus)))


def getFLaroundSeq(df,genome):
    donor_plus,donor_minus,acceptor_plus,acceptor_minus=flsite(df)
    seq_donor_plus=[aroundSeq(i,'+',genome) for i in donor_plus]
    seq_donor_minus=[aroundSeq(i,'-',genome) for i in donor_minus]
    seq_acceptor_plus=[aroundSeq(i,'+',genome) for i in acceptor_plus]
    seq_acceptor_minus=[aroundSeq(i,'-',genome) for i in acceptor_minus]
    
    seq_donor=[]
    seq_acceptor=[]
    seq_donor_pos=[]
    seq_acceptor_pos=[]
    td=seq_donor_plus+seq_donor_minus
    tdp=donor_plus+donor_minus
    ta=seq_acceptor_plus+seq_acceptor_minus
    tap=acceptor_plus+acceptor_minus
    for i in range(len(td)):
        e_td=td[i]
        e_tdp=tdp[i]
        if e_td[76:78]=='GT':
            seq_donor.append(e_td)
            seq_donor_pos.append(e_tdp)
    for i in range(len(ta)):
        e_ta=ta[i]
        e_tap=tap[i]
        if e_ta[73:75]=='AG':
            seq_acceptor.append(e_ta)
            seq_acceptor_pos.append(e_tap)
    return(seq_donor,seq_acceptor,seq_donor_pos,seq_acceptor_pos)
    
def filterDf(df,threshold=180):
    dist=df.end-df.start
    newdf=df.loc[(dist>threshold) & (df.strand!='.'),:]
    newdf.index=range(newdf.shape[0])
    return(newdf)
def filterDf_new(df,threshold=180):
    dist=df.end-df.start
    newdf=df.loc[(dist>threshold) & (df.strand!='.'),:]
    return(newdf)
def extendPredProb(df,strand,genome,donor_model,acceptor_model):
    donor_pos_list=[];acceptor_pos_list=[];donor_predict_list=[];acceptor_predict_list=[]
    for i in range(0,df.shape[0],1000):
        donor_pos,acceptor_pos,donor_predict,acceptor_predict=predProb(df.iloc[i:(i+1000)],strand,genome,donor_model,acceptor_model)
        donor_pos_list.extend(donor_pos)
        acceptor_pos_list.extend(acceptor_pos)
        donor_predict_list.extend(donor_predict)
        acceptor_predict_list.extend(acceptor_predict)
    return(donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list)
    
def predProb(df,strand,genome,donor_model,acceptor_model):
    donorposl,acceptorposl=df2pos(df,genome)
    donorposl_all=[]
    acceptorposl_all=[]
    for i in donorposl:
        for j in i:
            donorposl_all.append(j)
    for i in acceptorposl:
        for j in i:
            acceptorposl_all.append(j)            
    donorposl_all=list(set(donorposl_all))
    acceptorposl_all=list(set(acceptorposl_all))
    donorposl_all_pro=predictSpliceProb(donorposl_all,genome,donor_model,acceptor_model,strand=strand)
    acceptorposl_all_pro=predictSpliceProb(acceptorposl_all,genome,donor_model,acceptor_model,strand=strand,target='AG')
    donorposl_dict={}
    acceptorposl_dict={}
    for i,v in enumerate(donorposl_all):
        donorposl_dict[v]=donorposl_all_pro[i][0]
    for i,v in enumerate(acceptorposl_all):
        acceptorposl_dict[v]=acceptorposl_all_pro[i][0]
    donor_pos,donor_predict=getCanSite(donorposl,donorposl_dict)
    acceptor_pos,acceptor_predict=getCanSite(acceptorposl,acceptorposl_dict)
    return(donor_pos,acceptor_pos,donor_predict,acceptor_predict)
    
def df2pos(df,genome):
    donorposl=[]
    acceptorposl=[]
    for i in range(df.shape[0]):   
        each=df.iloc[i,:]
        chr=each['chr']
        start=each['start']
        end=each['end']
        strand=each['strand']
        donorpos,acceptorpos=site2pos(chr,start,end,strand,genome)
        donorposl.append(donorpos)
        acceptorposl.append(acceptorpos)
    return(donorposl,acceptorposl)
    
def site2pos(chr,start,end,strand,genome):
    donorpos,acceptorpos=pos2site(chr,start,end,strand,genome)
    #print(donorpos)
    donorpos=[chr+'|'+str(i) for i in donorpos]
    acceptorpos=[chr+'|'+str(i) for i in acceptorpos]
    return(donorpos,acceptorpos)
    
def pos2site(chr,start,end,strand,genome):
    seq=genome.sequence({'chr': chr, 'start':start, 'stop':end,'strand':strand}).upper()
    donorsite=seq2splice(seq)
    acceptorsite=seq2splice(seq,'AG')
    donorsite,acceptorsite=judgeSpliceAble(donorsite,acceptorsite)
    if len(donorsite)==0 or len(acceptorsite)==0:
        return([],[])
    if strand=='+':
        donorpos=[i+start for i in donorsite]
        acceptorpos=[i+start for i in acceptorsite]
    else:
        donorpos=[end-i for i in donorsite]
        acceptorpos=[end-i for i in acceptorsite]
    return(donorpos,acceptorpos)
    
def seq2splice(seq,tseq='GT',microExon=30):
    canpos=[]
    for i in range(microExon,len(seq)-microExon):
        pseq=seq[i:(i+2)]
        if seq[i:(i+2)]==tseq:
            if tseq=='GT':
                canpos.append(i-1)
            elif tseq=='AG':
                canpos.append(i+2) 
            else:
                return('Error')
    return(canpos)
    
def judgeSpliceAble(donorpos,acceptorpos):
    nd=[]
    na=[]
    if len(donorpos)==0 or len(acceptorpos)==0:
        return([],[])
    else:
        for i in donorpos:
            if i <acceptorpos[-1]:
                nd.append(i)
        for i in acceptorpos:
            if i>donorpos[0]:
                na.append(i)
        return(nd,na)
        
def predictSpliceProb(pos,genome,donor_model,acceptor_model,strand='+',target='GT'):
    if strand=='+':
        seq=[aroundSeq(i,'+',genome) for i in pos]
    else:
        seq=[aroundSeq(i,'-',genome) for i in pos]
    seq_onehot=torch.from_numpy(encodeSeqs(seq).astype(np.float32))
    if target=='GT':
        test_output=donor_model(seq_onehot[0:(0+10000),]).data.cpu().numpy()
        for i in range(10000,seq_onehot.shape[0],10000):
            #print(i)
            test_output =np.append(test_output,donor_model(seq_onehot[i:(i+10000),]).data.cpu().numpy(),axis=0)
        return(test_output)
    else:
        test_output=acceptor_model(seq_onehot[0:(0+10000),]).data.cpu().numpy()
        for i in range(10000,seq_onehot.shape[0],10000):
            #print(i)
            test_output =np.append(test_output,acceptor_model(seq_onehot[i:(i+10000),]).data.cpu().numpy(),axis=0)
        return(test_output)
        
def getCanSite(pos,pred,threshold=0.2):
    passList=[]
    predList=[]
    for each in pos:
        tmp=[]
        tmp2=[]
        for i in each:
            if pred[i]>threshold:
                tmp.append(i)
                tmp2.append(pred[i])
        passList.append(tmp)
        predList.append(tmp2)
    return(passList,predList)
    
def extendPredProb_dist(df,strand,genome,donor_model,acceptor_model,fl_fail,distSS):
    donor_pos_list=[];acceptor_pos_list=[];donor_predict_list=[];acceptor_predict_list=[]
    for i in range(0,df.shape[0],1000):
        donor_pos,acceptor_pos,donor_predict,acceptor_predict=predProb_dist(df.iloc[i:(i+1000),:],strand,genome,donor_model,acceptor_model,fl_fail.iloc[i:(i+1000),:],distSS)
        donor_pos_list.extend(donor_pos)
        acceptor_pos_list.extend(acceptor_pos)
        donor_predict_list.extend(donor_predict)
        acceptor_predict_list.extend(acceptor_predict)
    return(donor_pos_list,acceptor_pos_list,donor_predict_list,acceptor_predict_list)
    
def predProb_dist(df,strand,genome,donor_model,acceptor_model,fl_fail,distSS):
    donorposl,acceptorposl=df2pos_dist(df,genome,fl_fail,distSS)
    donorposl_all=[]
    acceptorposl_all=[]
    for i in donorposl:
        for j in i:
            donorposl_all.append(j)
    for i in acceptorposl:
        for j in i:
            acceptorposl_all.append(j)            
    donorposl_all=list(set(donorposl_all))
    acceptorposl_all=list(set(acceptorposl_all))
    if len(donorposl_all)>0:
        donorposl_all_pro=predictSpliceProb(donorposl_all,genome,donor_model,acceptor_model,strand=strand)
    if len(acceptorposl_all)>0:
        acceptorposl_all_pro=predictSpliceProb(acceptorposl_all,genome,donor_model,acceptor_model,strand=strand,target='AG')
    donorposl_dict={}
    acceptorposl_dict={}
    for i,v in enumerate(donorposl_all):
        donorposl_dict[v]=donorposl_all_pro[i][0]
    for i,v in enumerate(acceptorposl_all):
        acceptorposl_dict[v]=acceptorposl_all_pro[i][0]
    donor_pos,donor_predict=getCanSite(donorposl,donorposl_dict)
    acceptor_pos,acceptor_predict=getCanSite(acceptorposl,acceptorposl_dict)
    return(donor_pos,acceptor_pos,donor_predict,acceptor_predict)
    
def df2pos_dist(df,genome,fl_fail,distSS):
    donorposl=[]
    acceptorposl=[]
    for i in range(df.shape[0]):   
        each=df.iloc[i,:]
        chr=each['chr']
        start=each['start']
        end=each['end']
        strand=each['strand']
        each_fl=fl_fail.iloc[i,:]
        dSite=[int(j) for j in each_fl['donorSite'].split(',')]
        aSite=[int(j) for j in each_fl['acceptorSite'].split(',')]
        donorpos,acceptorpos=site2pos_dist(chr,start,end,strand,genome,dSite,aSite,distSS)
        donorposl.append(donorpos)
        acceptorposl.append(acceptorpos)
    return(donorposl,acceptorposl)
    
def site2pos_dist(chr,start,end,strand,genome,dSite,aSite,distSS):
    donorpos,acceptorpos=pos2site_dist(chr,start,end,strand,genome,dSite,aSite,distSS)
    #print(donorpos)
    donorpos=[chr+'|'+str(i) for i in donorpos]
    acceptorpos=[chr+'|'+str(i) for i in acceptorpos]
    return(donorpos,acceptorpos)
    
def pos2site_dist(chr,start,end,strand,genome,dSite,aSite,distSS):
    seq=genome.sequence({'chr': chr, 'start':start, 'stop':end,'strand':strand}).upper()
    donorsite=seq2splice(seq)
    acceptorsite=seq2splice(seq,'AG')
    donorsite,acceptorsite=judgeSpliceAble(donorsite,acceptorsite)
    if len(donorsite)==0 or len(acceptorsite)==0:
        return([],[])
    if strand=='+':
        donorpos=[i+start for i in donorsite]
        acceptorpos=[i+start for i in acceptorsite]
    else:
        donorpos=[end-i for i in donorsite]
        acceptorpos=[end-i for i in acceptorsite]
    donorpos_new=[]
    acceptorpos_new=[]
    for j in donorpos:
        for i in dSite:
            if (j>=(i-distSS)) and (j<=(i+distSS)):
                donorpos_new.append(j)
                break
    for j in acceptorpos:
        for i in aSite:
            if (j>=(i-distSS)) and (j<=(i+distSS)):
                acceptorpos_new.append(j)
                break
    return(list(set(donorpos_new)),list(set(acceptorpos_new)))
