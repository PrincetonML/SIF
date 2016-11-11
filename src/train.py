
import sys, os
from time import time
import random
import numpy as np

from params import params
import argparse

from theano import config
import lasagne
from sklearn.decomposition import TruncatedSVD

import data_io
from proj_model_sim import proj_model_sim
from proj_model_sentiment import proj_model_sentiment
import eval

##################################################
def str2bool(v):
    "utility function for parsing boolean arguments"
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def learner2bool(v):
    "utility function for parsing the argument for learning optimization algorithm"
    if v is None:
        return lasagne.updates.adam
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')


def get_pc(data, We, weight4ind, params):
    "Comput the principal component"

    def get_weighted_average(We, x, w):
        "Compute the weighted average vectors"
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, We.shape[1]))
        for i in xrange(n_samples):
            emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
        return emb

    for i in data:
        i[0].populate_embeddings(words)
        if not params.task == "sentiment":
            i[1].populate_embeddings(words)
    if params.task == "ent":
        (scores,g1x,g1mask,g2x,g2mask) = data_io.getDataEntailment(data)
        if params.weightfile:
            g1mask = data_io.seq2weight(g1x, g1mask, weight4ind)
    elif params.task == "sim":
        (scores,g1x,g1mask,g2x,g2mask) = data_io.getDataSim(data, -1)
        if params.weightfile:
            g1mask = data_io.seq2weight(g1x, g1mask, weight4ind)
    elif params.task == "sentiment":
        (scores,g1x,g1mask) = data_io.getDataSentiment(data)
        if params.weightfile:
            g1mask = data_io.seq2weight(g1x, g1mask, weight4ind)
    emb = get_weighted_average(We, g1x, g1mask)
    svd = TruncatedSVD(n_components=params.npc, n_iter=7, random_state=0)
    svd.fit(emb)
    return svd.components_

def train_util(model, train_data, dev, test, train, words, params):
    "utility function for training the model"
    start_time = time()
    try:
        for eidx in xrange(params.epochs):
            kf = data_io.get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:
                uidx += 1
                batch = [train_data[t] for t in train_index]
                # load the word ids
                for i in batch:
                    i[0].populate_embeddings(words)
                    if not params.task == "sentiment":
                        i[1].populate_embeddings(words)
                # load the data
                if params.task == "ent":
                    (scores,g1x,g1mask,g2x,g2mask) = data_io.getDataEntailment(batch)
                elif params.task == "sim":
                    (scores,g1x,g1mask,g2x,g2mask) = data_io.getDataSim(batch, model.nout)
                elif params.task == "sentiment":
                    (scores,g1x,g1mask) = data_io.getDataSentiment(batch)
                else:
                    raise ValueError('Task should be ent or sim.')
                # train
                if not params.task == "sentiment":
                    if params.weightfile:
                        g1mask = data_io.seq2weight(g1x, g1mask, params.weight4ind)
                        g2mask = data_io.seq2weight(g2x, g2mask, params.weight4ind)
                    cost = model.train_function(scores, g1x, g2x, g1mask, g2mask)
                else:
                    if params.weightfile:
                        g1mask = data_io.seq2weight(g1x, g1mask, params.weight4ind)
                    cost = model.train_function(scores, g1x, g1mask)
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                # undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[0].unpopulate_embeddings()
                    if not params.task == "sentiment":
                        i[1].representation = None
                        i[1].unpopulate_embeddings()
            # evaluate
            if params.task == "sim":
                dp,ds = eval.supervised_evaluate(model,words,dev,params)
                tp,ts = eval.supervised_evaluate(model,words,test,params)
                rp,rs = eval.supervised_evaluate(model,words,train,params)
                print "evaluation: ",dp,ds,tp,ts,rp,rs
            elif params.task == "ent" or params.task == "sentiment":
                ds = eval.supervised_evaluate(model,words,dev,params)
                ts = eval.supervised_evaluate(model,words,test,params)
                rs = eval.supervised_evaluate(model,words,train,params)
                print "evaluation: ",ds,ts,rs
            else:
                raise ValueError('Task should be ent or sim.')
            print 'Epoch ', (eidx+1), 'Cost ', cost
            sys.stdout.flush()
    except KeyboardInterrupt:
        print "Training interupted"
    end_time = time()
    print "total time:", (end_time - start_time)


##################################################
# initialize
random.seed(1)
np.random.seed(1)

# parse arguments
print sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("-LW", help="Lambda for word embeddings (normal training).", type=float)
parser.add_argument("-LC", help="Lambda for composition parameters (normal training).", type=float)
parser.add_argument("-batchsize", help="Size of batch.", type=int)
parser.add_argument("-dim", help="Size of input.", type=int)
parser.add_argument("-memsize", help="Size of classification layer.",
                    type=int)
parser.add_argument("-wordfile", help="Word embedding file.")
parser.add_argument("-layersize", help="Size of output layers in models.", type=int)
parser.add_argument("-updatewords", help="Whether to update the word embeddings")
parser.add_argument("-traindata", help="Training data file.")
parser.add_argument("-devdata", help="Training data file.")
parser.add_argument("-testdata", help="Testing data file.")
parser.add_argument("-nonlinearity", help="Type of nonlinearity in projection and DAN model.",
                    type=int)
parser.add_argument("-nntype", help="Type of neural network.")
parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
parser.add_argument("-minval", help="Min rating possible in scoring.", type=int)
parser.add_argument("-maxval", help="Max rating possible in scoring.", type=int)
parser.add_argument("-clip", help="Threshold for gradient clipping.",type=int)
parser.add_argument("-eta", help="Learning rate.", type=float)
parser.add_argument("-learner", help="Either AdaGrad or Adam.")
parser.add_argument("-task", help="Either sim, ent, or sentiment.")
parser.add_argument("-weightfile", help="The file containing the weights for words; used in weighted_proj_model_sim.")
parser.add_argument("-weightpara", help="The parameter a used in computing word weights.", type=float)
parser.add_argument("-npc", help="The number of principal components to use.", type=int, default=0)
args = parser.parse_args()

params = params()
params.LW = args.LW
params.LC = args.LC
params.batchsize = args.batchsize
params.hiddensize = args.dim
params.memsize = args.memsize
params.wordfile = args.wordfile
params.nntype = args.nntype
params.layersize = args.layersize
params.updatewords = str2bool(args.updatewords)
params.traindata = args.traindata
params.devdata = args.devdata
params.testdata = args.testdata
params.nntype = args.nntype
params.epochs = args.epochs
params.learner = learner2bool(args.learner)
params.task = args.task
params.weightfile = args.weightfile
params.weightpara = args.weightpara
params.npc = args.npc

if args.eta:
    params.eta = args.eta
params.clip = args.clip
if args.clip:
    if params.clip == 0:
        params.clip = None
params.minval = args.minval
params.maxval = args.maxval
if args.nonlinearity:
    if args.nonlinearity == 1:
        params.nonlinearity = lasagne.nonlinearities.linear
    if args.nonlinearity == 2:
        params.nonlinearity = lasagne.nonlinearities.tanh
    if args.nonlinearity == 3:
        params.nonlinearity = lasagne.nonlinearities.rectify
    if args.nonlinearity == 4:
        params.nonlinearity = lasagne.nonlinearities.sigmoid

# load data
(words, We) = data_io.getWordmap(params.wordfile)
if args.task == "sim" or args.task == "ent":
    train_data = data_io.getSimEntDataset(params.traindata,words,params.task)
elif args.task == "sentiment":
    train_data = data_io.getSentimentDataset(params.traindata,words)
else:
    raise ValueError('Task should be ent, sim, or sentiment.')

# load weight
if params.weightfile:
    word2weight = data_io.getWordWeight(params.weightfile, params.weightpara)
    params.weight4ind = data_io.getWeight(words, word2weight)
    print 'word weights computed using parameter a=' + str(params.weightpara)
else:
    params.weight4ind = []
if params.npc > 0:
    params.pc = get_pc(train_data, We, params.weight4ind, params)
else:
    params.pc = []

# load model
model = None
if params.nntype == 'proj':
    model = proj_model_sim(We, params)
elif params.nntype == 'proj_sentiment':
    model = proj_model_sentiment(We, params)
else:
    "Error no type specified"

# train
train_util(model, train_data, params.devdata, params.testdata, params.traindata, words, params)
