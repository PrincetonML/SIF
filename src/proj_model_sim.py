import numpy as np
import theano
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer

class proj_model_sim(object):

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

        if params.maxval:
            self.nout = params.maxval - params.minval + 1

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        if params.npc > 0:
            pc = theano.shared(np.asarray(params.pc, dtype = config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_average = lasagne_average_layer([l_emb, l_mask])
        l_out = lasagne.layers.DenseLayer(l_average, params.layersize, nonlinearity=params.nonlinearity)

        embg1 = lasagne.layers.get_output(l_out, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_out, {l_in:g2batchindices, l_mask:g2mask})

        if params.npc <= 0:
            print "#pc <=0, do not remove pc"
        elif params.npc == 1:
            print "#pc == 1"
            proj1 =  embg1.dot(pc.transpose())
            proj2 =  embg2.dot(pc.transpose())
            embg1 = embg1 - theano.tensor.outer(proj1, pc)
            embg2 = embg2 - theano.tensor.outer(proj2, pc)
        else:
            print "#pc > 1"
            proj1 =  embg1.dot(pc.transpose())
            proj2 =  embg2.dot(pc.transpose())
            embg1 = embg1 - theano.tensor.dot(proj1, pc)
            embg2 = embg2 - theano.tensor.dot(proj2, pc)

        g1_dot_g2 = embg1*embg2
        g1_abs_g2 = abs(embg1-embg2)

        lin_dot = lasagne.layers.InputLayer((None, params.layersize))
        lin_abs = lasagne.layers.InputLayer((None, params.layersize))
        l_sum = lasagne.layers.ConcatLayer([lin_dot, lin_abs])
        l_sigmoid = lasagne.layers.DenseLayer(l_sum, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        if params.task == "sim":
            l_softmax = lasagne.layers.DenseLayer(l_sigmoid, self.nout, nonlinearity=T.nnet.softmax)
            X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})
            Y = T.log(X)

            cost = scores*(T.log(scores) - Y)
            cost = cost.sum(axis=1)/(float(self.nout))

            prediction = 0.
            i = params.minval
            while i<= params.maxval:
                prediction = prediction + i*X[:,i-1]
                i += 1
        elif params.task == "ent":
            l_softmax = lasagne.layers.DenseLayer(l_sigmoid, 3, nonlinearity=T.nnet.softmax)
            X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})

            cost = theano.tensor.nnet.categorical_crossentropy(X,scores)

            prediction = T.argmax(X, axis=1)
        else:
            raise ValueError('Params.task not set correctly.')

        self.network_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0)  # do not include the word embedding as network parameters
        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.scoring_function = theano.function([g1batchindices, g2batchindices,
                             g1mask, g2mask],prediction)
        self.cost_function = theano.function([scores, g1batchindices, g2batchindices,
                             g1mask, g2mask], cost)

        #updates
        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g2batchindices,
                             g1mask, g2mask], cost, updates=updates)