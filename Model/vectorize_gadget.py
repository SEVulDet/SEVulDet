import os
import warnings
from gensim.models import Word2Vec
import numpy

warnings.filterwarnings("ignore")

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}', '!'
}


class GadgetVectorizer:

    def __init__(self, vector_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.embeddings = 0

    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1

            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 3])
                w = []
                i += 3
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1

            else:
                w.append(line[i])
                i += 1

        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))

    @staticmethod
    def tokenize_gadget(gadget):
        tokenized = []
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            tokenized += tokens
        return tokenized

    def add_gadget(self, gadget):
        tokenized_gadget = GadgetVectorizer.tokenize_gadget(gadget)
        # print(len(tokenized_gadget), tokenized_gadget)
        self.gadgets.append(tokenized_gadget)

    def vectorize(self, gadget):
        tokenized_gadget = GadgetVectorizer.tokenize_gadget(gadget)
        vectors = numpy.zeros(shape=(max(len(tokenized_gadget), 10), self.vector_length))
        for i in range(len(tokenized_gadget)):
            vectors[i] = self.embeddings[tokenized_gadget[i]]
        return vectors

    def train_model(self, filename):
        base = os.path.splitext(os.path.basename(filename))[0]
        word2vec_filename = base + "_word2vec.model"
        if os.path.exists(word2vec_filename):
            model = Word2Vec.load(word2vec_filename)
        else:
            model = Word2Vec(self.gadgets, min_count=1, size=self.vector_length, sg=1)
            model.save(word2vec_filename)
        self.embeddings = model.wv
        del model
        del self.gadgets
