# SIF

This is the code for [the paper](https://openreview.net/forum?id=SyK00v5xx) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings".

The code is written in python and requires numpy, scipy, pickle, sklearn, theano and the lasagne library. 
Some functions/classes are based on the [code](https://github.com/jwieting/iclr2016) of John Wieting for the paper "Towards Universal Paraphrastic Sentence Embeddings". The example data sets are also preprocessed using the code there.

## Get started
To get started, cd into the directory examples/ and run demo.sh. It downloads the pretrained GloVe word embeddings, and then runs the scripts: 
* sim_sif.py sim_tfidf.py are for the textual similarity tasks
* supervised_sif_proj.sh is for the supervised tasks 

Check these files to see the options.

## Source code
The code is separated into 3 parts:

* textual similarity tasks: involves data_io.py, eval.py, and sim_algo.py. data_io provides the code for reading the data, eval is for evaluating the performance, and sim_algo provides the code for our sentence embedding algorithm.
* supervised tasks: involves data_io.py, eval.py, train.py, proj_model_sim.py, and proj_model_sentiment.py. train provides the entry for training the models (proj_model_sim is for the similarity and entailment tasks, and proj_model_sentiment is for the sentiment task). Check train.py to see the options.
* utilities: includes lasagne_average_layer.py, params.py, and tree.py. These provides utility functions/classes for the above two parts. 

## References
For technical details, see [the paper](https://openreview.net/forum?id=SyK00v5xx).
```
@article{arora2016asimple, 
	author = {Sanjeev Arora and Yingyu Liang and Tengyu Ma}, 
	title = {A Simple but Tough-to-Beat Baseline for Sentence Embeddings}, 
	year = {2016}
}
```