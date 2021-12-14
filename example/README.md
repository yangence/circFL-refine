## Train CNN models to predict donor and acceptor sites

### Required files
```
bedFile=hg19_huamn_circ/human_tier1.bed # included in example/hg19_huamn_circ.tar.gz
genome=hg19.fa
```

### Training
```
circFL-refine train -f $bedFile -g $genome
```

## Evaluate identified full-length circRNA

### Required files
```
bedFile=circFL_refine_example/fl_iso.bed # included in example/circFL_refine_example.tar.gz
acceptor=circFL_refine_example/train/acceptor_model.net # included in example/circFL_refine_example.tar.gz
donor=circFL_refine_example/train/donor_model.net # included in example/circFL_refine_example.tar.gz
genome=hg19.fa
```

### Evaluation
```
circFL-refine evaluate  -f $bedFile -a $acceptor -d $donor -g $genome
```

## Correct mistaken circRNA isoforms

### Required files
```
bedFile=circFL_refine_example/evaluate/fl_evaluate.txt # included in example/circFL_refine_example.tar.gz
acceptor=circFL_refine_example/train/acceptor_model.net # included in example/circFL_refine_example.tar.gz
donor=circFL_refine_example/train/donor_model.net # included in example/circFL_refine_example.tar.gz
genome=hg19.fa
```

### Correction
```
circFL-refine correct -f $bedFile -a $acceptor -d $donor -g $genome
```