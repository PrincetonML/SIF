import pickle, sys
sys.path.append('../src')
import data_io, sim_algo, eval, params

## run
wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    '../data/glove.840B.300d.txt'  # need to download it first
    ]

weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
weightparas = [-1, 1e-3]#[-1,1e-1,1e-2,1e-3,1e-4]
rmpcs = [0,1]# [0,1,2]

params = params.params()
parr4para = {}
sarr4para = {}
for wordfile in wordfiles:
    (words, We) = data_io.getWordmap(wordfile)
    for weightpara in weightparas:
        word2weight = data_io.getWordWeight(weightfile, weightpara)
        weight4ind = data_io.getWeight(words, word2weight)
        for rmpc in rmpcs:
            print 'word vectors loaded from %s' % wordfile
            print 'word weights computed from %s using parameter a=%f' % (weightfile, weightpara)
            params.rmpc = rmpc
            print 'remove the first %d principal components' % rmpc
            ## eval just one example dataset
            parr, sarr = eval.sim_evaluate_one(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
            ## eval all datasets; need to obtained datasets from John Wieting (https://github.com/jwieting/iclr2016)
            # parr, sarr = eval.sim_evaluate_all(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
            paras = (wordfile, weightfile, weightpara, rmpc)
            parr4para[paras] = parr
            sarr4para[paras] = sarr

## save results
save_result = False #True
result_file = 'result/sim_sif.result'
comment4para = [ # need to align with the following loop
    ['word vector files', wordfiles], # comments and values,
    ['weight parameters', weightparas],
    ['remove principal component or not', rmpcs]
]
if save_result:
    with open(result_file, 'w') as f:
        pickle.dump([parr4para, sarr4para, comment4para] , f)

