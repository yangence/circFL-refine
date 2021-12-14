# circFL-refine: a tool for full-length circRNA isoform construction based on convolutional neural network

## Introduction

circFL-refine is a circRNA analysis pipeline to evaluate and correct full-length structure of circRNA isoform

## Availability

circFL-refine is a free software, which can be downloaded from https://github.com/yangence/circFL-refine

Well trained human and mouse donor/acceptor models are available in https://github.com/yangence/circFL-refine/tree/master/model (human_acceptor_model_0.net, human_donor_model_0.net, mouse_acceptor_model_0.net, mouse_donor_model_0.net).

## Softwares and packages dependencies
Python 3.x.x and corresponding versions of numpy, pandas, torch, pyfasta, scikit-learn, docopt.


## Installation
Install latest release from pip
```
pip install circFL-refine
```

Install latest release from source codes
```
git clone https://github.com/yangence/circFL-refine.git
cd circFL-refine/script
pip install -r requirements.txt
python setup.py install
```

## Required files:

Users can prepare the external files under the following instructions:

Indexed genome fasta file

```
samtools faidx $genome
```

## Examples:
https://github.com/yangence/circFL-refine/tree/master/example

## Usage
```
Usage: circFL-refine <command> [options]
Command:
    train             Train CNN models to predict donor and acceptor sites
    evaluate          Evaluate full-length circRNA isoforms
    correct           Correct mistaken circRNA isoforms
```

### train
```
Usage: circFL-refine train -f circ -g genome [-l lines] [-e epoch] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     BED format file contain full-length circRNA information.
    -g genome                   Fasta file of genome.
    -l lines=K                  Only use first K lines for training, e.g. 30000.
    -e epoch                    Number of epoch for training [default: 1].
    -o output                   Output dir [default: circFL_refine].
```

### evaluate
```
Usage: circFL-refine evaluate -f circ -g genome [-d donor_model] [-a acceptor_model] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     BED format file contain full-length circRNA information.
    -g genome                   Fasta file of genome.
    -d donor_model              Donor model file, output of train commond.
    -a acceptor_model           Acceptor model file, output of train commond.
    -o output                   Output dir [default: circFL_refine].
```

### correct
```
Usage: circFL-refine correct -f circ -g genome -d donor_model -a acceptor_model [-e mistake] [-i dist] [-o output]

Options:
    -h --help                   Show help message.
    -v --version                Show version.
    -f circ                     Evaluation result of full-length circRNA, output of evaluate commond.
    -g genome                   Fasta file of genome.
    -d donor_model              Donor model file, output of train commond.
    -a acceptor_model           Acceptor model file, output of train commond.
    -e mistake=k                Isoforms with less than k mistaken splice site to be corrected [default: 1].
    -i dist                     Maximum distance of nearby predicted splice site [default: 20].
    -o output                   Output dir [default: circFL_refine].
```

## Output files
| File name         |  Details | 
|   :---            | ---        |
| acceptor_model.net       | acceptor splice site prediction model from `train`|
| donor_model.net          | donor splice site prediction model from `train` |
| fl_evaluate.txt          | `evaluate` results |
| fl_correct.detail         | details of `correct` results |
| fl_correct.bed            | corrected BED file from `correct` |

### fl_evaluate.txt
| No. | Column name     |  Details | 
|:---:|   :---          | ---        |
|  1  | chrom          | chromosome |
|  2  | chromStart           | start coordinate of circRNA |
|  3  | chromEnd             | end coordinate of circRNA |
|  4  | name           | isoform ID |
|  5  | score             | 0 |
|  6  | strand             | strand of circRNA |
|  7  | thickStart         | . |
|  8  | thickEnd      | . |
|  9  | itemRgb        | . |
|  10 | blockCount           | number of exons in circRNA |
|  11 | blockSizes         | block size of each exon |
|  12 | blockStarts        | block start of each exon |
|  13 | donorSite    | corrdinate of donor splice site |
|  14 | donorMotif   | motif of donor splice site |
|  15 | donorPredict          | prediction score of donor splice site |
|  16 | donorPredictMin        | minium prediction score of donor splice site |
|  17 | acceptorSite       | corrdinate of acceptor splice site |
|  18 | acceptorMotif          | motif of acceptor splice site |
|  19 | acceptorPredict   | prediction score of acceptor splice site |
|  20 | acceptorPredictMin          | minium prediction score of acceptor splice site |

Isoforms with donorPredictMin>0.5 and acceptorPredictMin>0.5 are considered as evaluate passed.

### fl_correct.detail
| No. | Column name     |  Details | 
|:---:|   :---          | ---        |
|  1  | chrom          | chromosome |
|  2  | chromStart           | start coordinate of circRNA |
|  3  | chromEnd             | end coordinate of circRNA |
|  4  | name           | isoform ID |
|  5  | score             | 0 |
|  6  | strand             | strand of circRNA |
|  7  | thickStart         | . |
|  8  | thickEnd      | . |
|  9  | itemRgb        | . |
|  10 | blockCount           | number of exons in circRNA |
|  11 | blockSizes         | block size of each exon |
|  12 | blockStarts        | block start of each exon |
|  13 | donorSite    | corrdinate of donor splice site |
|  14 | donorMotif   | motif of donor splice site |
|  15 | donorPredict          | prediction score of donor splice site |
|  16 | donorPredictMin        | minium prediction score of donor splice site |
|  17 | acceptorSite       | corrdinate of acceptor splice site |
|  18 | acceptorMotif          | motif of acceptor splice site |
|  19 | acceptorPredict   | prediction score of acceptor splice site |
|  20 | acceptorPredictMin          | minium prediction score of acceptor splice site |
|  21 | correct_donorSite          | corrdinate of corrected donor splice site |
|  22 | correct_acceptorSite        | corrdinate of corrected acceptor splice site |
|  23 | dN       | number of mistaken donor splice site |
|  24 | aN          | number of mistaken acceptor splice site |
|  25 | dM   | maximum distance of corrected donor splice site to original site |
|  26 | aM          | maximum distance of corrected acceptor splice site to original site |

### fl_correct.bed
| No. | Column name     |  Details | 
|:---:|   :---          | ---        |
|  1  | chrom          | chromosome |
|  2  | chromStart           | start coordinate of circRNA |
|  3  | chromEnd             | end coordinate of circRNA |
|  4  | name           | isoform ID |
|  5  | score             | 0 |
|  6  | strand             | strand of circRNA |
|  7  | thickStart         | . |
|  8  | thickEnd      | . |
|  9  | itemRgb        | . |
|  10 | blockCount           | number of exons in circRNA |
|  11 | blockSizes         | block size of each exon |
|  12 | blockStarts        | block start of each exon |

Copyright (C) 2021 Zelin Liu (zlliu@bjmu.edu.cn). See the [LICENSE](https://github.com/yangence/circFL-refine/blob/master/LICENSE) file for license rights and limitations.
