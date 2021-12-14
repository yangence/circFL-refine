#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author: Zelin Liu
email: zlliu@bjmu.edu.cn
license: GPL3
detail: A deep learning method to predict full-length circRNA
"""
from docopt import docopt
import sys,warnings
from circFL_refine.version import __version__
warnings.filterwarnings("ignore")
#__version__=0.01

helpInfo_old = '''
Usage: circFL-refine <command> [options]
Command:
    train             Train CNN models to predict donor and acceptor sites
    evaluate          Evaluate identified full-length circRNA
    construct         Construct full-length circRNA
    correct           Correct mistaken circRNA isoforms
'''
helpInfo = '''
Usage: circFL-refine <command> [options]
Command:
    train             Train CNN models to predict donor and acceptor sites
    evaluate          Evaluate full-length circRNA isoforms
    correct           Correct mistaken circRNA isoforms
'''

def main():
    command_log = 'circFL-refine parameters: ' + ' '.join(sys.argv)
    if len(sys.argv) == 1:
        sys.exit(helpInfo)
    elif sys.argv[1] == '--version' or sys.argv[1] == '-v':
        sys.exit(__version__)
    elif sys.argv[1] == 'train':
        import circFL_refine.train as train
        train.train(docopt(train.__doc__, version=__version__))
    elif sys.argv[1] == 'evaluate':
        import circFL_refine.evaluate as evaluate
        evaluate.evaluate(docopt(evaluate.__doc__, version=__version__))
    elif sys.argv[1] == 'construct':
        import circFL_refine.construct as construct
        construct.construct(docopt(construct.__doc__, version=__version__))
    elif sys.argv[1] =='correct':
        import circFL_refine.correct as correct
        correct.correct(docopt(correct.__doc__, version=__version__))
    else:
        sys.exit(helpInfo)

if __name__ == '__main__':
    main()