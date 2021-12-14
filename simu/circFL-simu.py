'''
Usage: circFL-simu.py -g genome -a annotation -o outDir -e errorRate -n numRC [--min minLengh] [--max maxLength] [-c maxCoverage]  [-i circNum]

Options:
    -h --help                   Show help message.
    -g genome                   Fasta file of genome.
    -a annotation               GTF file.
    -o outDir                   Output dir.
    -e errorRate                Error rate.
    -n numRC                    Number of rolling circles.
    --min minLengh              Minimum length of circRNA isoform [default: 400].
    --max maxLength             Maximum length of circRNA isoform [default: 600].
    -c maxCoverage              Maximum coverage of full-length circRNA isoform [default: 10].
    -i circNum                  Number of simulated circRNAs [default: 10000].
'''
import pandas as pd,numpy as np,pysam,os,sys,gzip
from docopt import docopt

options=docopt(__doc__)
errRate_all=float(options['-e'])
if (errRate_all<0) or (errRate_all>1):
    os.exit('Paratemers errRate is out of range!!!')
errRate=[0,0,0]
errRate[0]=errRate_all/2 # mismatch
errRate[1]=errRate_all/4 # insertion
errRate[2]=errRate_all/4 # deletion
    
outDir=options['-o']
numRC=float(options['-n'])

minCirc=int(options['--min'])
maxCirc=int(options['--max'])
COV_NUM=int(options['-c'])
circNum=int(options['-i'])

circCov=[np.random.randint(1,COV_NUM) for i in range(circNum)]
minReadLength=200

if not os.path.exists(outDir):
    os.mkdir(outDir)
gtfFile=options['-a']
faFile=options['-g']

def readGTFfile(fileName):
    if not os.path.exists(fileName):
        sys.exit('ERROR: %s is not exist!!!' % fileName)
    try:
        tabixfile = pysam.TabixFile(fileName)
        return(tabixfile)
    except:
        sys.exit('ERROR: make sure %s is sorted and tabix indexed!!!' % fileName)


gtfFile=readGTFfile(gtfFile)
faFile=pysam.FastaFile(faFile)

gene_dict={}
gene_chr={}
gene_strand={}
for gtf in gtfFile.fetch(parser=pysam.asGTF()):
    if gtf.feature=='gene':
        gene_chr[gtf.gene_id]=gtf.contig
        gene_strand[gtf.gene_id]=gtf.strand
    if gtf.feature=='exon':
        if gene_dict.__contains__(gtf.gene_id):
            if gene_dict[gtf.gene_id].__contains__(gtf.transcript_id):
                gene_dict[gtf.gene_id][gtf.transcript_id].append([gtf.start,gtf.end])
            else:
                gene_dict[gtf.gene_id][gtf.transcript_id]=[[gtf.start,gtf.end]]
        else:
            gene_dict[gtf.gene_id]={}
            gene_dict[gtf.gene_id][gtf.transcript_id]=[[gtf.start,gtf.end]]
            

exon_maxNum={}
gene_can3=[]
gene_can5=[]
for i in gene_dict.keys():
    exon_maxNum[i]=0
    for j in gene_dict[i].keys():
        exon_maxNum[i]=max(exon_maxNum[i],len(gene_dict[i][j]))
    if exon_maxNum[i]>2:
        gene_can3.append(i)
    if exon_maxNum[i]>4:
        gene_can5.append(i)

gene_can=gene_can3

def getSeq(arr,contig,genome):
    seq=''
    for i in arr:
        seq+=genome.fetch(contig,i[0],i[1],)
    return(seq)

def revCom(seq):
    return(seq[::-1].upper().replace('A','t').replace('T','a').replace('G','c').replace('C','g').upper())

def getErr(read1,errNum):
    base2base={'A':['T','G','C'],'T':['A','G','C'],'G':['T','A','C'],'C':['T','G','A'],'N':['A','G','T']}
    readLen=len(read1)
    pos=np.random.choice(len(read1),errNum,replace=False).tolist()
    read1=list(read1)
    for i in pos:
        read1[i]=base2base[read1[i]][np.random.randint(3)]
    return(''.join(read1))
def getErr_insert(read1,errNum,dist=[2,5]):
    readLen=len(read1)
    pos=np.random.choice(len(read1),errNum,replace=False).tolist()
    read1=list(read1)
    for i in pos:
        read1[i]=''.join(list(np.random.choice(['A','C','G','T'],np.random.randint(dist[0],dist[1]))))
    return(''.join(read1))
def getErr_del(read1,errNum,dist=[2,5]):
    readLen=len(read1)
    pos=np.random.choice(len(read1),errNum,replace=False).tolist()
    read1=list(read1)
    for i in pos:
        tmp=np.random.randint(dist[0],dist[1])
        for j in range(i,min(i+tmp,readLen)):
            read1[j]=''
    return(''.join(read1))

def simulate_circ(cov,readLen,seq,errRate,out1,id,strand='u'):
    seqLen_raw=len(seq)
    #numRead=max(int(cov*seqLen_raw/readLen),1)
    numRead=cov
    dupNum=int(readLen/seqLen_raw)+2
    seq=''.join([seq for i in range(dupNum)])
    seqLen=dupNum*seqLen_raw
    errBaseNum=int(errRate[0]*numRead*readLen)
    errBaseNum_insert=int(errRate[1]*numRead*readLen)
    errBaseNum_del=int(errRate[2]*numRead*readLen)
    eachReaderr_dict={}
    eachReaderr_insert_dict={}
    eachReaderr_del_dict={}
    for i in range(errBaseNum):
        tmpErr=np.random.randint(numRead)
        if eachReaderr_dict.__contains__(tmpErr):
            eachReaderr_dict[tmpErr]+=1
        else:
            eachReaderr_dict[tmpErr]=1
    for i in range(errBaseNum_insert):
        tmpErr=np.random.randint(numRead)
        if eachReaderr_insert_dict.__contains__(tmpErr):
            eachReaderr_insert_dict[tmpErr]+=1
        else:
            eachReaderr_insert_dict[tmpErr]=1
    for i in range(errBaseNum_del):
        tmpErr=np.random.randint(numRead)
        if eachReaderr_del_dict.__contains__(tmpErr):
            eachReaderr_del_dict[tmpErr]+=1
        else:
            eachReaderr_del_dict[tmpErr]=1
    i=0
    while i<numRead:
        #print('i=%d;numRead=%d' % (i,numRead))
        startLoci=np.random.randint(max(seqLen-readLen+1,0))
        start2Loci=startLoci+readLen
        juncNum=int(start2Loci / seqLen_raw) - int(startLoci / seqLen_raw)
        if juncNum<1:
            continue
         
        if strand=='first' or strand=='f' or strand=='F':
            read1=seq[startLoci:(startLoci+readLen)].upper()
        else:
            if np.random.randint(2):
                read1=seq[startLoci:(startLoci+readLen)].upper()
            else:
                read1=revCom(seq[startLoci:(startLoci+readLen)])
        if eachReaderr_dict.__contains__(i):
            if eachReaderr_dict[i]>=readLen:
                i+=1
                continue
                
            read1=getErr(read1,eachReaderr_dict[i])
        if eachReaderr_insert_dict.__contains__(i):
            if eachReaderr_insert_dict[i]>=readLen:
                i+=1
                continue
                
            read1=getErr_insert(read1,eachReaderr_insert_dict[i])
        if eachReaderr_del_dict.__contains__(i):
            if eachReaderr_del_dict[i]>=readLen:
                i+=1
                continue
                
            read1=getErr_del(read1,eachReaderr_del_dict[i])
        readLen=len(read1)
        #if readLen<300: There is different with simu_nano.py
        #    i+=1
         #   continue
            
        readName='@'+id+':'+str(juncNum)+':'+str(i)
        quality=''.join(['F' for i in range(readLen)])
        out1.write(bytes("%s\n%s\n+\n%s\n" %(readName+' length='+str(readLen), read1,quality),encoding='ASCII'))
        i+=1

def getMaxIntronLen(x):
    tmpMax=0
    for i in range(0,len(x)-1):
        each1=x[i]
        each2=x[i+1]
        tmp=each2[0]-each1[1]
        if tmp>tmpMax:
            tmpMax=tmp
    return(tmpMax)
        
def getAllCirc(exon,maxLen=5000,maxIntronLen=200000):
    exonArr=[]
    for i in range(len(exon)):
        for j in range(i,len(exon)):
            tmp=exon[i:(j+1)]
            exonLen=sum([(i[1]-i[0]) for i in tmp])
            if exonLen<maxLen:
                if len(tmp)>1:
                    maxIntron=getMaxIntronLen(tmp)
                    if maxIntron<maxIntronLen:                        
                        exonArr.append(tmp)
                else:
                    exonArr.append(tmp)
    return(exonArr)
        
def exonTrans(exon):
    newExon=','.join([str(i[0]+1)+'-'+str(i[1]) for i in exon])
    return(newExon)

# simulate circFL-seq
read1_file=os.path.join(outDir,'test_1.fq.gz')
info_file=os.path.join(outDir,'test_info.txt')
read1_out=gzip.open(read1_file,'wb')
info_out=open(info_file,'w')
count=0
circ_list=[]
exonDict={}
readID=0
for i in gene_can:
    tran_dict=gene_dict[i]
    contig=gene_chr[i]
    exonDict[contig]={}
    for tranID in tran_dict.keys():
        exon_arr=tran_dict[tranID]
        circAll=[]
        circ_exon=[exon_arr[x] for x in range(1,len(exon_arr)-1)] # full exon
        circ_exon_pass=[]
        for j in circ_exon:
            if j[1]-j[0]>30:
                circ_exon_pass.append(j)
        circAll=getAllCirc(circ_exon_pass)
        circAll_pass=[]
        for p in circAll:
            if exonDict[contig].__contains__(str(p)):
                continue
            else:
                circAll_pass.append(p)
                exonDict[contig][str(p)]=1
        for p in circAll_pass:
            circStrand=gene_strand[i]
            circSeq=getSeq(p,contig,faFile)
            circSeq_len=len(circSeq)
            if circSeq_len>maxCirc or circSeq_len<minCirc:
                continue
            if circStrand=='-':
                circSeq=revCom(circSeq)
            motif=''
            circID=contig+':'+str(p[0][0]+1)+'|'+str(p[-1][1])
            circStart=','.join([str(i[0]+1) for i in p])
            circEnd=','.join([str(i[1]) for i in p])
            info_out.write("%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n" % (i,tranID,circID,readID,circStart,circEnd,exonTrans(p),circSeq,circCov[readID],circSeq_len,motif))

            for m in range(circCov[readID]):
                readLen=max(int(numRC*circSeq_len)+np.random.randint(0,100),minReadLength+np.random.randint(0,100))
                simulate_circ(1,readLen,circSeq,errRate,read1_out,'03:%s:%d:%d' % (str(i)[0:15],readID,m))
            readID+=1
            break
    #count+=1
        #print(readID)
        if readID==circNum:
            break
        break # one gene one circRNA
    #print(count)
    if readID==circNum:
        break
read1_out.close()
info_out.close()
