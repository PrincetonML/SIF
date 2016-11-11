import data_io

class tree(object):

    def __init__(self, phrase, words):
        self.phrase = phrase
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            self.embeddings.append(data_io.lookupIDX(words,i))

    def unpopulate_embeddings(self):
        self.embeddings = []