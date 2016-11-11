from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import data_io


################################################
## for textual similarity tasks
################################################
def sim_getCorrelation(We,words,f, weight4ind, scoring_function, params):
    f = open(f,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = data_io.getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = data_io.prepare_data(seq1)
    x2,m2 = data_io.prepare_data(seq2)
    m1 = data_io.seq2weight(x1, m1, weight4ind)
    m2 = data_io.seq2weight(x2, m2, weight4ind)
    scores = scoring_function(We,x1,x2,m1,m2, params)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def sim_evaluate_all(We, words, weight4ind, scoring_function, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["MSRpar2012",
            "MSRvid2012",
            "OnWN2012",
            "SMTeuro2012",
            "SMTnews2012", # 4
            "FNWN2013",
            "OnWN2013",
            "SMT2013",
            "headline2013", # 8
            "OnWN2014",
            "deft-forum2014",
            "deft-news2014",
            "headline2014",
            "images2014",
            "tweet-news2014", # 14
            "answer-forum2015",
            "answer-student2015",
            "belief2015",
            "headline2015",
            "images2015",    # 19
            "sicktest",
            "twitter",
            "JHUppdb",
            "anno-dev",
            "anno-test"]

    for i in farr:
        p,s = sim_getCorrelation(We, words, prefix+i, weight4ind, scoring_function, params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j in zip(farr, parr):
        s += "%30s %10f\n" % (i, j)

    n = sum(parr[0:5]) / 5.0
    s += "%30s %10f \n" % ("2012-average ", n)

    n = sum(parr[5:9]) / 4.0
    s += "%30s %10f \n" % ("2013-average ", n)

    n = sum(parr[9:15]) / 6.0
    s += "%30s %10f \n" % ("2014-average ", n)

    n = sum(parr[15:20]) / 5.0
    s += "%30s %10f \n" % ("2015-average ", n)

    print s

    return parr, sarr


def sim_evaluate_one(We, words, weight4ind, scoring_function, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["MSRpar2012"]

    for i in farr:
        p,s = sim_getCorrelation(We, words, prefix+i, weight4ind, scoring_function, params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j in zip(farr, parr):
        s += "%30s %10f\n" % (i, j)
    print s

    return parr, sarr


################################################
## for supervised tasks
################################################
def getCorrelation(model,words,f, params=[]):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = data_io.getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = data_io.prepare_data(seq1)
    x2,m2 = data_io.prepare_data(seq2)
    if params and params.weightfile:
        m1 = data_io.seq2weight(x1, m1, params.weight4ind)
        m2 = data_io.seq2weight(x2, m2, params.weight4ind)
    scores = model.scoring_function(x1,x2,m1,m2)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def acc(preds,scores):
    golds = []
    for n,i in enumerate(scores):
        p = -1
        i=i.strip()
        if i == "CONTRADICTION":
            p = 0
        elif i == "NEUTRAL":
            p = 1
        elif i == "ENTAILMENT":
            p = 2
        else:
            raise ValueError('Something wrong with data...')
        golds.append(p)
    #print confusion_matrix(golds,preds)
    return accuracy_score(golds,preds)

def accSentiment(preds,scores):
    golds = []
    for n,i in enumerate(scores):
        p = -1
        i=i.strip()
        if i == "0":
            p = 0
        elif i == "1":
            p = 1
        else:
            raise ValueError('Something wrong with data...')
        golds.append(p)
    return accuracy_score(golds,preds)

def getAcc(model,words,f, params=[]):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = i[2]
        X1, X2 = data_io.getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = data_io.prepare_data(seq1)
            x2,m2 = data_io.prepare_data(seq2)
            if params and params.weightfile:
                m1 = data_io.seq2weight(x1, m1, params.weight4ind)
                m2 = data_io.seq2weight(x2, m2, params.weight4ind)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = data_io.prepare_data(seq1)
        x2,m2 = data_io.prepare_data(seq2)
        if params and params.weightfile:
            m1 = data_io.seq2weight(x1, m1, params.weight4ind)
            m2 = data_io.seq2weight(x2, m2, params.weight4ind)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc(preds,golds)

def getAccSentiment(model,words,f, params=[]):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; score = i[1]
        X1 = data_io.getSeq(p1,words)
        seq1.append(X1)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = data_io.prepare_data(seq1)
            if params and params.weightfile:
                m1 = data_io.seq2weight(x1, m1, params.weight4ind)
            scores = model.scoring_function(x1,m1)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = data_io.prepare_data(seq1)
        if params and params.weightfile:
            m1 = data_io.seq2weight(x1, m1, params.weight4ind)
        scores = model.scoring_function(x1,m1)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return accSentiment(preds,golds)

def supervised_evaluate(model,words,file,params):
    if params.task == "sim":
        p,s = getCorrelation(model,words,file, params)
        return p,s
    elif params.task == "ent":
        s = getAcc(model,words,file, params)
        return s
    elif params.task == "sentiment":
        s = getAccSentiment(model,words,file, params)
        return s
    else:
        raise ValueError('Task should be ent, sim, or sentiment')
