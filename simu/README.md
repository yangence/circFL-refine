## Simulate full-length circRNA nanopore seqencing data

## Required Files:

Users can prepare the external files under the following instructions:

1) Indexed genome fasta file

```bash
samtools faidx $genome
```

2) Tabix indexed gene annotation GTF file

```bash
grep -v '#' $gtf |sort -k 1,1 -k 4,4n |bgzip >sort.gtf.gz
tabix sort.gtf.gz
```

## Usage
```
Usage: circFL-simu.py -g genome -a annotation -o outDir -e errorRate -n numRC [--min minLengh] [--max maxLength] [-c maxCoverage]  [-i circNum]

Options:
    -h --help                   Show help message.
    -o outDir                   Output dir.
    -e errorRate                Error rate.
    -n numRC                    Number of rolling circles.
    --min minLengh              Minimum length of circRNA isoform [default: 400].
    --max maxLength             Maximum length of circRNA isoform [default: 600].
    -c maxCoverage              Maximum coverage of full-length circRNA isoform [default: 10].
    -g genome                   Fasta file of genome.
    -a annotation               GTF file.
    -i circNum                  Number of simulated circRNAs [default: 10000].
```

## Example
```
genome=hg19.fa
gtf=sort.gtf.gz
NUM_COV=10
mkdir simu_10
for i in {0.1,0.05}; do for j in {1.5,2,3};do for m in {1..9};do n=`expr $m + 1`;echo "python circFL-simu.py -o simu_${NUM_COV}/compare_simu_${i}_${j}_${m}00_${n}00_${NUM_COV} -e ${i} -n ${j} --min ${m}00 --max ${n}00 -c ${NUM_COV} -g $genome -a $gtf ";done ;done;done
```
