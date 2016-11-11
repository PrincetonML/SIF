import numpy as np
import theano
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer

class proj_model_sentiment(object):

    def getRegTerm(self, params, We, initial_We):
        l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
        if params.updatewords:
            return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
        else:
            return l2

    def getTrainableParams(self, params):
        if params.updatewords:
            return self.all_params
        else:
            return self.network_params

    def __init__(self, We_initial, params):
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        if params.npc > 0:
            pc = theano.shared(np.asarray(params.pc, dtype = config.floatX))

        g1batchindices = T.imatrix()
        g1mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_average = lasagne_average_layer([l_emb, l_mask])
        l_out = lasagne.layers.DenseLayer(l_average, params.layersize, nonlinearity=params.nonlinearity)
        embg = lasagne.layers.get_output(l_out, {l_in:g1batchindices, l_mask:g1mask})
        if params.npc <= 0:
            print "#pc <=0, do not remove pc"
        elif params.npc == 1:
            print "#pc == 1"
            proj =  embg.dot(pc.transpose())
            embg = embg - theano.tensor.outer(proj, pc)
        else:
            print "#pc > 1"
            proj =  embg.dot(pc.transpose())
            embg = embg - theano.tensor.dot(proj, pc)

        l_in2 = lasagne.layers.InputLayer((None, params.layersize))
        l_sigmoid = lasagne.layers.DenseLayer(l_in2, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, 2, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {l_in2:embg})
        cost = T.nnet.categorical_crossentropy(X,scores)
        prediction = T.argmax(X, axis=1)

        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0) # do not include the word embedding as network parameters
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg)
        self.scoring_function = theano.function([g1batchindices, g1mask],prediction)
        self.cost_function = theano.function([scores, g1batchindices, g1mask], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g1mask], cost, updates=updates)
