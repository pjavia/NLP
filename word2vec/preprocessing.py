import theano as th
import numpy as np
import cPickle 

par = (5,(128, 1), 'datasetSentences.txt')
# Give window size odd number please... lol makes the coding easy...


class word2vec:
    
    def __init__(self, params):
        
        # Change parameters accordingly
        
        self.window_size, self.embeddings_dim, self.raw_data = params

        self.count_word2vec = dict() # word : count
        self.order_word2vec = dict() # key : word
        self.raw_train_data = dict()
        self.embeddings = dict()
        self.vocabulary = 0
        
    # Preprocessing based on data avaialable add appropriate definition definition but don't mess up domain and range of function.
    def sentimental_analysis(self):
        raw_lines = []
        with open(self.raw_data) as f:
            
            for i in f:
                raw_lines.append(i)
        del raw_lines[0]
        order = 0
        for j in range(0,len(raw_lines)):
            key = int(raw_lines[j].split('\t')[0])
            raw_sentence = raw_lines[j].split('\t')[1]
            #print raw_sentence
            processed_sentence = raw_sentence.split(' ')
            del processed_sentence[-1]
            processed_sentence.append('.')
            sentence = processed_sentence
            #print sentence
            self.raw_train_data.update({key:sentence})
            for k in sentence:
                if k in self.count_word2vec.keys():
                    self.count_word2vec[k] += 1
                else:
                    self.order_word2vec.update({order:k})
                    order += 1
                    self.count_word2vec.update({k:0})
    	#print self.order_word2vec
       	#print self.count_word2vec
        data = {'words_with_keys':self.order_word2vec, 'words_with_count':self.count_word2vec, 'train_data':self.raw_train_data}
        with open('raw_data','wb') as fp:
            cPickle.dump(data, fp)
                
    def prepare_data_for_word2vec(self):
        
        with open('raw_data','rb') as fp:
            f = cPickle.load(fp)
        num_to_words = f.get('words_with_keys')
        self.vocabulary = (num_to_words.keys())
    	for i in num_to_words.keys():
    		self.embeddings.update({i:np.random.randn(self.embeddings_dim[0], self.embeddings_dim[1])})

    	for j in range(0, self.window_size):

	def distribution(self):

		return neg_sample

    	


            
w = word2vec(par)
#w.sentimental_analysis()
w.prepare_data_for_word2vec()    
        
                
                        
                 
            
                
        
        # The data should be as a line. Also, the apostrophe s should be like this 's seperated from the real thing.



