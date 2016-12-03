echo 'not updating word vectors, removing 0 pc, layersize 2400'
echo 'GloVe vectors'
echo 'similarity'
# weight 1e-2
    sh train.sh -wordfile ../data/glove.840B.300d.txt -updatewords False -dim 300 -traindata ../data/sicktrain -devdata ../data/sickdev -testdata ../data/sicktest -layersize 2400 -nntype proj -nonlinearity 1 -epochs 20 -minval 1 -maxval 5 -task sim -batchsize 25 -LW 1e-05 -LC 1e-06 -memsize 50 -learner adam -eta 0.001 -weightfile ../auxiliary_data/enwiki_vocab_min200.txt -weightpara 1e-2
		
echo 'entailment'
# weight 1e-2
    sh train.sh -wordfile ../data/glove.840B.300d.txt -updatewords False -dim 300 -traindata ../data/sicktrain-ent -devdata ../data/sickdev-ent -testdata ../data/sicktest-ent -layersize 2400 -nntype proj -nonlinearity 1 -epochs 20 -task ent -batchsize 25 -LW 1e-05 -LC 1e-06 -memsize 150 -learner adagrad -eta 0.05 -weightfile ../auxiliary_data/enwiki_vocab_min200.txt -weightpara 1e-2
	
echo 'sentiment'
# weight 1e-2
    sh train.sh -wordfile ../data/glove.840B.300d.txt -updatewords False -dim 300 -traindata ../data/sentiment-train -devdata ../data/sentiment-dev -testdata ../data/sentiment-test -layersize 2400 -nntype proj_sentiment -nonlinearity 1 -epochs 20 -task sentiment -batchsize 25 -LW 1e-05 -LC 1e-06 -memsize 150 -learner adagrad -eta 0.05 -weightfile ../auxiliary_data/enwiki_vocab_min200.txt -weightpara 1e-3  
	
