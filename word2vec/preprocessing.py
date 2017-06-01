import numpy as np
import cPickle
import random


par = (2, (128, 1), 'datasetSentences.txt')


# Give window size odd number please... lol makes the coding easy...


class word2vec:
    def __init__(self, params):

        # Change parameters accordingly

        self.window_size, self.embeddings_dim, self.raw_data = params

        self.count_word2vec = dict()  # word : count
        self.order_word2vec = dict()  # key : word
        self.word_key = dict()
        self.raw_train_data = dict()
        self.embeddings = dict()
        self.vocabulary = 0
        self.train_pair = []
        self.numeral_pairs = []
        self.train_list = []
        self.codomain = []

    # Preprocessing based on data avaialable add appropriate definition definition but don't mess up domain and range of function.
    def sentimental_analysis(self):
        raw_lines = []
        with open(self.raw_data) as f:

            for i in f:
                raw_lines.append(i)
        del raw_lines[0]
        order = 0
        for j in range(0, len(raw_lines)):
            key = int(raw_lines[j].split('\t')[0])
            raw_sentence = raw_lines[j].split('\t')[1]
            # print raw_sentence
            processed_sentence = raw_sentence.split(' ')
            del processed_sentence[-1]
            processed_sentence.append('.')
            sentence = processed_sentence
            # print sentence
            self.raw_train_data.update({key: sentence})
            for k in sentence:
                if k in self.count_word2vec.keys():
                    self.count_word2vec[k] += 1
                else:
                    self.order_word2vec.update({order: k})
                    self.word_key.update({k:order})
                    order += 1
                    self.count_word2vec.update({k: 1})
        #print self.order_word2vec
        #print self.count_word2vec
        #print self.word_key

    def prepare_data_for_word2vec(self):

        #print self.order_word2vec.keys()
        for i in self.order_word2vec.keys():
            self.embeddings.update({i: np.random.rand(self.embeddings_dim[0], self.embeddings_dim[1])})
            cc = self.count_word2vec.get((self.order_word2vec.get(i)))
            for y_ in range(0, cc):
                self.codomain.append(i)
        #print self.codomain

        for k in self.raw_train_data.values():
            for l in range(0, len(k)):
                words = []
                for m in range(1, self.window_size+1):
                    if l - m >= 0:
                        self.train_pair.append((k[l],k[l-m]))
                        self.numeral_pairs.append((self.word_key.get(k[l]),self.word_key.get(k[l-m])))
                        words.append(self.word_key.get(k[l-m]))

                    if l + m <= len(k)-1:
                        self.train_pair.append((k[l], k[l + m]))
                        self.numeral_pairs.append((self.word_key.get(k[l]), self.word_key.get(k[l+m])))
                        words.append(self.word_key.get(k[l+m]))

                self.train_list.append((self.word_key.get(k[l]), words))

    def train(self):

        neta = -0.01

        for epoch in range(0, 50):
            print 'epoch', epoch

            for i in self.train_list:
                loss = 0
                W_i, labels = i
                neg_labels = self.negative_samples(labels)
                h = self.embeddings.get(W_i)
                for j in labels:

                    v = self.embeddings.get(j)

                    v += neta*self.sigmoid(np.dot(np.transpose(self.embeddings.get(j)), h))*h

                    #print v , 'v'



                    loss += -1.0*np.log(np.abs(self.sigmoid(np.dot(np.transpose(self.embeddings.get(j)), h))*h))

                    self.embeddings[j] = v

                for k in neg_labels:
                    v = self.embeddings.get(k)
                    if k in labels:
                        v += neta * self.sigmoid(np.dot(np.transpose(self.embeddings.get(j)), h)) * h

                        loss += -1.0 * np.log(np.abs(self.sigmoid(np.dot(np.transpose(self.embeddings.get(k)), h))*h))
                    else:
                        v += neta * neta *(self.sigmoid(np.dot(np.transpose(self.embeddings.get(k)), h))-1.0)*h
                        loss += -1.0 * np.log(np.abs(self.sigmoid(-1.0*np.dot(np.transpose(self.embeddings.get(k)), h))*h))

                    self.embeddings[k] = v
                print 'loss', np.sum(loss)



    def negative_samples(self, num_to_remove):


        show = random.sample(self.codomain, 20)

        return show

    def sigmoid(self, x):


            return 1.0 /(1.0 + np.exp(-x))











w = word2vec(par)
w.sentimental_analysis()
w.prepare_data_for_word2vec()
w.train()

with open('trained_dump','wb') as fp:
    cPickle.dump(w.embeddings,fp)
    cPickle.dump(w.order_word2vec,fp)
    cPickle.dump(w.word_key,fp)
    cPickle.dump(w.count_word2vec,fp)








# The data should be as a line. Also, the apostrophe s should be like this 's seperated from the real thing.
