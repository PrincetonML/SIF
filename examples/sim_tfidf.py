import pickle, sys
sys.path.append('../src')
import data_io, sim_algo, eval, params


## run
wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    '../data/glove.840B.300d.txt'  # need to download it first
    ]
rmpcs = [0,1]

comment4para = [ # need to align with the following loop
    ['word vector files', wordfiles], # comments and values,
    ['remove principal component or not', rmpcs]
]

params = params.params()
parr4para = {}
sarr4para = {}
for wordfile in wordfiles:
    (words, We) = data_io.getWordmap(wordfile)
    weight4ind = data_io.getIDFWeight(wordfile)
    for rmpc in rmpcs:
        print 'word vectors loaded from %s' % wordfile
        print 'word weights computed from idf'
        params.rmpc = rmpc
        print 'remove the first %d principal components' % rmpc
        # eval just one example dataset
        parr, sarr = eval.sim_evaluate_one(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
        ## eval all datasets; need to obtained datasets from John Wieting (https://github.com/jwieting/iclr2016)
        # parr, sarr = eval.sim_evaluate_all(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
        paras = (wordfile, rmpc)
        parr4para[paras] = parr
        sarr4para[paras] = sarr

## save result
save_result = False # True
result_file = 'result/sim_tfidf.result'
if save_result:
    with open(result_file, 'w') as f:
        pickle.dump([parr4para, sarr4para, comment4para] , f)
