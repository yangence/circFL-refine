'''
Usage: circFL-refine train -f circ -g genome [-l lines] [-e epoch] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     BED format file contain full-length circRNA information.
    -g genome                   Fasta file of genome.
    -l lines=K                  Only use first K lines for training, e.g. 30000.
    -e epoch                    Number of epoch for training [default: 1].
    -o output                   Output dir [default: circFL_refine].
'''

# circFL-refine.py train -f fl_pass.bed -g /media/data4/lzl/genome/hg19/hg19.fa
from .genericFun import *
import sys,time
import pandas as pd
import numpy as np
import os
import pyfasta
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, classification_report,recall_score  ,accuracy_score,precision_score,f1_score,auc,roc_curve,average_precision_score

def fl2exon(df):
    exonKey={}
    for i in range(df.shape[0]):
        #if i % 1000 ==0:
        #    print(i)
        each=df.iloc[i,:]
        chr=each[0]
        start=each[1]
        end=each[2]
        eLen=[int(i) for i in each[10].split(',') if i!='']
        eStart=[int(i) for i in each[11].split(',') if i!='']
        
        for j in range(len(eLen)):
            
            subStart=start+eStart[j]
            subEnd=subStart+eLen[j]
            eachKey=each[3]+'-'+chr+'|'+str(subStart)+'|'+str(subEnd)
            exonKey[eachKey]=[chr,subStart,subEnd,chr+'|'+str(subStart)+'|'+str(subEnd),0,each[5],0]

    return(exonKey)



def randomSeq(chr,start,end,strand,tseq,around=75):
    seq=genome.sequence({'chr': chr, 'start':start+around, 'stop':end-around,'strand':strand}).upper()
    canpos=[]
    for i in range(len(seq)):
        pseq=seq[i:(i+2)]
        if seq[i:(i+2)]==tseq:
            if strand=='+':
                if tseq=='GT':
                    canpos.append(i+start+around-1)
                else:
                       canpos.append(i+start+around+2) 
            else:
                if tseq=='GT':
                    canpos.append(end-around-i+1)
                else:
                    canpos.append(end-around-i-2)
    return(canpos)
    
def getBGpos(df):
    background_acceptor_pos_plus=[];background_acceptor_pos_minus=[];background_donor_pos_plus=[];background_donor_pos_minus=[]
    for i in range(df.shape[0]):
        each=df.iloc[i,:]
        chr=each[0]
        start=each[1]
        end=each[2]
        strand=each[5]
        if end-start>151:
            tmpSeq=randomSeq(chr,start,end,strand,'AG')
            tmpPos=[chr+'|'+str(i) for i in tmpSeq]
            if strand=='+':
                background_acceptor_pos_plus.extend(tmpPos)
            else:
                background_acceptor_pos_minus.extend(tmpPos)
            tmpSeq=randomSeq(chr,start,end,strand,'GT')
            tmpPos=[chr+'|'+str(i) for i in tmpSeq]
            if strand=='+':
                background_donor_pos_plus.extend(tmpPos)
            else:
                background_donor_pos_minus.extend(tmpPos)
    background_acceptor_pos_plus_adjust=list(set(background_acceptor_pos_plus)-set(all_junc_key))
    background_donor_pos_plus_adjust=list(set(background_donor_pos_plus)-set(all_junc_key))
    background_acceptor_pos_minus_adjust=list(set(background_acceptor_pos_minus)-set(all_junc_key))
    background_donor_pos_minus_adjust=list(set(background_donor_pos_minus)-set(all_junc_key))
    
    bg_seq_donor_plus=[aroundSeq(i,'+',genome) for i in background_donor_pos_plus_adjust]
    bg_seq_donor_minus=[aroundSeq(i,'-',genome) for i in background_donor_pos_minus_adjust]
    bg_seq_acceptor_plus=[aroundSeq(i,'+',genome) for i in background_acceptor_pos_plus_adjust]
    bg_seq_acceptor_minus=[aroundSeq(i,'-',genome) for i in background_acceptor_pos_minus_adjust]
    
    bg_seq_donor=bg_seq_donor_plus+bg_seq_donor_minus
    bg_seq_acceptor=bg_seq_acceptor_plus+bg_seq_acceptor_minus
    return(bg_seq_donor,bg_seq_acceptor)

def getspecificity(real,predict):
    tn=0
    fp=0
    for i,value in enumerate(real):
        if value==0:
            if predict[i]==0:
                tn+=1
            else:
                fp+=1
    return(tn/(tn+fp))

def getTrainResult(model,val_x,val_y,test_x,test_y):
    # evaluation
    val_output=model(val_x[0:(0+10000),]).data.cpu().numpy()
    for i in range(10000,val_x.shape[0],10000):
        val_output =np.append(val_output,model(val_x[i:(i+10000),]).data.cpu().numpy(),axis=0)

    phase_label=val_y.data.cpu().numpy()
    auc = roc_auc_score(phase_label[:,0], val_output[:,0])
    accuracy = accuracy_score(phase_label[:,0],  [ round(i) for i in val_output[:,0]])
    recall=recall_score(phase_label[:,0],  [ round(i) for i in val_output[:,0]]) # is sensitivity
    precision = precision_score(phase_label[:,0],  [ round(i) for i in val_output[:,0]])
    f1=f1_score(phase_label[:,0],  [ round(i) for i in val_output[:,0]])
    auPRC=average_precision_score(phase_label[:,0], val_output[:,0])
    specificity=getspecificity(phase_label[:,0],  [ round(i) for i in val_output[:,0]])
    print("Validation set result accuracy:{:.4f} recall:{:.4f} specificity:{:.4f} precision:{:.4f} f1:{:.4f} auROC:{:.4f} auPRC {:.4f}".format(accuracy,recall,specificity,precision,f1,auc,auPRC,))

    test_output=model(test_x[0:(0+10000),]).data.cpu().numpy()
    for i in range(10000,test_x.shape[0],10000):
        test_output =np.append(test_output,model(test_x[i:(i+10000),]).data.cpu().numpy(),axis=0)

    phase_label=test_y.data.cpu().numpy()
    auc = roc_auc_score(phase_label[:,0], test_output[:,0])
    accuracy = accuracy_score(phase_label[:,0],  [ round(i) for i in test_output[:,0]])
    recall=recall_score(phase_label[:,0],  [ round(i) for i in test_output[:,0]])
    precision = precision_score(phase_label[:,0],  [ round(i) for i in test_output[:,0]])
    f1=f1_score(phase_label[:,0],  [ round(i) for i in test_output[:,0]])
    auPRC=average_precision_score(phase_label[:,0], test_output[:,0])
    specificity=getspecificity(phase_label[:,0],  [ round(i) for i in test_output[:,0]])
    print("Test set result accuracy:{:.4f} recall:{:.4f} specificity:{:.4f} precision:{:.4f} f1:{:.4f} auROC:{:.4f} auPRC {:.4f}".format(accuracy,recall,specificity,precision,f1,auc,auPRC,))


def train(options):
    EPOCH = int(options['-e'])
    BATCH_SIZE = 100
    LR = 0.001
    global genome,all_junc_key
    fl_file=options['-f']
    genomeFile=options['-g']
    outDir=options['-o']

    plog('Check genome file')
    genome=readFaFile(genomeFile)
    plog('Check circRNA file')
    fl_pass=readBedFile(fl_file)
    
    fl_pass=fl_pass.loc[fl_pass.iloc[:,0].isin(list(genome.keys())),:].copy()
    if options['-l']:
        lines=int(options['-l'])
        fl_pass=fl_pass.iloc[0:lines,:]
    if fl_pass.shape[0]<100:
        sys.exit('ERROR: you have %s circRNAs. Make sure you have more than 100 known circRNAs!!!' % fl_pass.shape[0])

    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if outDir[-1]!='/':
        outPrefix=outDir+'/'
    else:
        outPrefix=outDir
    train_outPrefix=outPrefix+'train/'
    train_tmp_outPrefix=train_outPrefix+'tmp/'
    createDir(outPrefix);createDir(train_outPrefix)
    fl_all_exon=fl2exon(fl_pass)
    fl_all_exon_df=pd.DataFrame(fl_all_exon.values())
    fl_all_exon_df=fl_all_exon_df.drop_duplicates(3)
    all_junc_key1=fl_all_exon_df.iloc[:,0]+'|'+(fl_all_exon_df.iloc[:,1]+1).map(str)
    all_junc_key2=fl_all_exon_df.iloc[:,0]+'|'+fl_all_exon_df.iloc[:,2].map(str)
    all_junc_key=set(list(all_junc_key1.tolist()+all_junc_key2.tolist()))
    
    plog('Prepare true donor/acceptor sites')
    seq_donor,seq_acceptor,seq_donor_pos,seq_acceptor_pos=getFLaroundSeq(fl_pass,genome)
    bg_seq_donor,bg_seq_acceptor=getBGpos(fl_pass)
    
    seq_donor_onehot=encodeSeqs(seq_donor).astype(np.float32)
    seq_acceptor_onehot=encodeSeqs(seq_acceptor).astype(np.float32)
    seq_donor_N=seq_donor_onehot.shape[0]
    seq_acceptor_N=seq_acceptor_onehot.shape[0]
    
    plog('Prepare false donor/acceptor sites')
    bg_donor_onehot=encodeSeqs(bg_seq_donor).astype(np.float32)
    bg_acceptor_onehot=encodeSeqs(bg_seq_acceptor).astype(np.float32)
    bg_donor_N=bg_donor_onehot.shape[0]
    bg_acceptor_N=bg_acceptor_onehot.shape[0]
    # 80% for training, 10% for validation, 10% for test
    # considering the unbalance of number of real and background, we selected the proportion respectively
    plog('Train donor sites')
    seq_train_index=np.random.choice(range(seq_donor_N),int(seq_donor_N*0.8),replace=False)
    seq_val_test_index=list(set(range(seq_donor_N))-set(seq_train_index))
    seq_val_index=np.random.choice(seq_val_test_index,int(seq_donor_N*0.1))
    seq_test_index=list(set(seq_val_test_index)-set(seq_val_index))
    
    ratioNum=1
    
    bg_train_index=np.random.choice(range(bg_donor_N),int(ratioNum*seq_donor_N*0.8),replace=False)
    bg_val_test_index=list(set(range(bg_donor_N))-set(bg_train_index))
    bg_val_index=np.random.choice(bg_val_test_index,int(ratioNum*seq_donor_N*0.1))
    
    bg_test_index=np.random.choice(list(set(bg_val_test_index)-set(bg_val_index)),int(ratioNum*seq_donor_N*0.1))
    seq_train_sample_N=len(seq_train_index)
    seq_test_sample_N=len(seq_test_index)
    seq_val_sample_N=len(seq_val_index)
    bg_train_sample_N=len(bg_train_index)
    bg_test_sample_N=len(bg_test_index)
    bg_val_sample_N=len(bg_val_index)
    
    print("train_sample_N:{} test_sample_N:{} validation_sample_N:{}".format(seq_train_sample_N+bg_train_sample_N,
                                                                         seq_test_sample_N+bg_test_sample_N,
                                                                         seq_val_sample_N+bg_val_sample_N))

    # train samples
    train_x=np.r_[seq_donor_onehot[seq_train_index],bg_donor_onehot[bg_train_index]]
    train_x = torch.from_numpy(train_x)
    train_y=[[1] for i in range(seq_train_sample_N)]+[[0] for i in range(bg_train_sample_N)]
    train_y = torch.Tensor(train_y)
    # validation samples
    val_x=np.r_[seq_donor_onehot[seq_val_index],bg_donor_onehot[bg_val_index]]
    val_x = torch.from_numpy(val_x)
    val_y=[[1] for i in range(seq_val_sample_N)]+[[0] for i in range(bg_val_sample_N)]
    val_y = torch.Tensor(val_y)
    # test samples
    test_x=np.r_[seq_donor_onehot[seq_test_index],bg_donor_onehot[bg_test_index]]
    test_x = torch.from_numpy(test_x)
    test_y=[[1] for i in range(seq_test_sample_N)]+[[0] for i in range(bg_test_sample_N)]
    test_y = torch.Tensor(test_y)
    
    donor_myModel=Splicenet(151,1).to("cpu")
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    
    #optimizer = torch.optim.SGD(myModel.parameters(), lr=LR)
    optimizer = torch.optim.Adam(donor_myModel.parameters(), lr=LR, betas=(0.9, 0.99))
    #loss_fun
    loss_func = nn.BCELoss()
    
    donor_myModel.train()
    #training loop
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            output = donor_myModel(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # evaluation
    plog('Evaluation model performance on validation and test sets')
    getTrainResult(donor_myModel,val_x,val_y,test_x,test_y)
    torch.save(donor_myModel.state_dict(),train_outPrefix+'donor_model.net')
    
    plog('Train acceptor sites')
    seq_train_index=np.random.choice(range(seq_acceptor_N),int(seq_acceptor_N*0.8),replace=False)
    seq_val_test_index=list(set(range(seq_acceptor_N))-set(seq_train_index))
    seq_val_index=np.random.choice(seq_val_test_index,int(seq_acceptor_N*0.1))
    seq_test_index=list(set(seq_val_test_index)-set(seq_val_index))

    bg_train_index=np.random.choice(range(bg_acceptor_N),int(ratioNum*seq_acceptor_N*0.8),replace=False)
    bg_val_test_index=list(set(range(bg_acceptor_N))-set(bg_train_index))
    bg_val_index=np.random.choice(bg_val_test_index,int(ratioNum*seq_acceptor_N*0.1))

    bg_test_index=np.random.choice(list(set(bg_val_test_index)-set(bg_val_index)),int(ratioNum*seq_acceptor_N*0.1))

    seq_train_sample_N=len(seq_train_index)
    seq_test_sample_N=len(seq_test_index)
    seq_val_sample_N=len(seq_val_index)
    bg_train_sample_N=len(bg_train_index)
    bg_test_sample_N=len(bg_test_index)
    bg_val_sample_N=len(bg_val_index)

    print("train_sample_N:{} test_sample_N:{} validation_sample_N:{}".format(seq_train_sample_N+bg_train_sample_N,
                                                                             seq_test_sample_N+bg_test_sample_N,
                                                                             seq_val_sample_N+bg_val_sample_N))

    # train samples
    train_x=np.r_[seq_acceptor_onehot[seq_train_index],bg_acceptor_onehot[bg_train_index]]
    train_x = torch.from_numpy(train_x)

    train_y=[[1] for i in range(seq_train_sample_N)]+[[0] for i in range(bg_train_sample_N)]
    train_y = torch.Tensor(train_y)

    # validation samples
    val_x=np.r_[seq_acceptor_onehot[seq_val_index],bg_acceptor_onehot[bg_val_index]]
    val_x = torch.from_numpy(val_x)
    val_y=[[1] for i in range(seq_val_sample_N)]+[[0] for i in range(bg_val_sample_N)]
    val_y = torch.Tensor(val_y)
    # test samples
    test_x=np.r_[seq_acceptor_onehot[seq_test_index],bg_acceptor_onehot[bg_test_index]]
    test_x = torch.from_numpy(test_x)
    test_y=[[1] for i in range(seq_test_sample_N)]+[[0] for i in range(bg_test_sample_N)]
    test_y = torch.Tensor(test_y)
    acceptor_myModel=Splicenet(151,1).to("cpu")
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    optimizer = torch.optim.Adam(acceptor_myModel.parameters(), lr=LR, betas=(0.9, 0.99))
    #loss_fun
    loss_func = nn.BCELoss()
    acceptor_myModel.train()
    #training loop
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            output = acceptor_myModel(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # evaluation
    plog('Evaluation model performance on validation and test sets')
    getTrainResult(acceptor_myModel,val_x,val_y,test_x,test_y)

    
    torch.save(acceptor_myModel.state_dict(),train_outPrefix+'acceptor_model.net')
    plog('All done!!!')
