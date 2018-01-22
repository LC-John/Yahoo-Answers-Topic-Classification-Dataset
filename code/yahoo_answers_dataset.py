# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:10:11 2018

@author: DrLC
"""

import pickle, gzip
import random
import numpy

import time

class Yahoo_Answers_Dataset(object):
    
    '''
    Yahoo! Answers Topic Classification Dataset
    
    The original dataset has 1,400,000 pieces of data for training, and
    60,000 pieces of data for testing. All question-answer pairs are
    limited within 10 topics, which is also the classification label for
    each piece of data.
    
    Make sure that you call this class after pre-processing the scv
    raw data. What you need is a test-set .pkl.gz file, a train-set .pkl.gz
    file, a dictionary .pkl.gz file.
    
    We concatate the question title and content as the input question, and
    try to predict the topic label. We select 55,133 from the test-set and
    1,296,335 from train-set, which all have a question lenght lower than
    100.
    
    This Class is the wrapped Yahoo! Answers dataset.
    
    Parameters:
        
        dict_size: the vocabulary number. The model would only read those
        frequently shown-up words, and see "UNKNOWN" when a not frequently
        word shows up.
        
        valid_ratio: valid-set size / original train-set size. We need valid
        -set to make sure that the model is not over-fitting, so a splitting
        on train-set is nessesary.
        
        max_len: max length of all sequences in train/test-set. As said, we
        selected questions with length under 100, and the default max_len is
        100 of course.
        
        test_path: pre-processed test-set path.
        
        train_path: pre-processed train-set path.
        
        dict_path: vocabulary frequency table generated from pre-processed
        dataset.
        
        rand_seed: random seed.
        
        debug: debug verbose flag. Default as False (no-verbose).
    '''
    
    def __init__(self, dict_size=10000, valid_ratio=0.1, max_len=100,
                 test_path="../dataset/yahoo_answers_test.pkl.gz",
                 train_path="../dataset/yahoo_answers_train.pkl.gz",
                 dict_path="../dataset/yahoo_answers_dict.pkl.gz",
                 rand_seed=1234, debug=False):
        
        self.__rand_seed = rand_seed
        random.seed(self.__rand_seed)
        numpy.random.seed(self.__rand_seed)
        
        self.__dict_size = dict_size
        self.__max_len = max_len
        self.__debug = debug
        if self.__debug:
            print ("vocabulary# = %d" % self.__dict_size)
            print ("max seq len = %d" % self.__max_len)
            print ("valid# / (valid# + train#) = %f" % valid_ratio)
        
        # load test-set
        if self.__debug:
            s = time.time()
        self.__test = {"seq": None, "label": None}
        f = gzip.open(test_path, "rb")
        d = pickle.load(f)
        self.__test["seq"] = d["text"]
        self.__test["label"] = d["label"]
        f.close()
        if self.__debug:
            e = time.time()
            print ("Test-set loaded!")
            print ("\tTest# = %d" % len(self.__test["label"]))
            print ("\tTime cost = %.3f sec" % (e-s))

        # load & split train/valid-set
        if self.__debug:
            s = time.time()
        self.__train = {"seq": None, "label": None}
        self.__valid = {"seq": None, "label": None}
        f = gzip.open(train_path, "rb")
        d = pickle.load(f)
        split_idx = int(valid_ratio*len(d["label"]))
        self.__valid["seq"] = d["text"][:split_idx]
        self.__valid["label"] = d["label"][:split_idx]
        self.__train["seq"] = d["text"][split_idx:]
        self.__train["label"] = d["label"][split_idx:]
        f.close()
        if self.__debug:
            e = time.time()
            print ("Train-set (along with valid-set) loaded!")
            print ("\tTrain# = %d" % len(self.__train["label"]))
            print ("\tValid# = %d" % len(self.__valid["label"]))
            print ("\tTime cost = %.3f sec" % (e-s))
        
        # load label dictionary
        self.__label_dictionary = {}
        for idx in range(len(d["lookup_table"])):
            self.__label_dictionary[idx+1] = d["lookup_table"][idx]
        if self.__debug:
            print ("Label dictionary loaded!")

        # load vocabulary dictionary
        # dict key is a word, and dict value is the index of the one-hot vector
        if self.__debug:
            s = time.time()
        self.__dictionary = {}
        f = gzip.open(dict_path, "rb")
        d = pickle.load(f)
        for idx in range(min(len(d), dict_size)):
            self.__dictionary[d[idx][0]] = idx
        f.close()
        if self.__debug:
            e = time.time()
            print ("Word token dictionary loaded!")
            print ("\tTime cost = %.3f sec" % (e-s))
        # generate reverse vocabulary dictionary
        # dict key is an index, and dict value is the word
        if self.__debug:
            s = time.time()
        self.__rev_dictionary = {v:k for k,v in self.__dictionary.items()}
        if self.__debug:
            e = time.time()
            print ("Reverse word token dictionary generated!")
            print ("\tTime cost = %.3f sec" % (e-s))
        
        # generate an epoch randomly
        if self.__debug:
            s = time.time()
        self.__av = random.sample(range(len(self.__train["label"])),
                                  len(self.__train["label"]))
        if self.__debug:
            print ("Epoch generated!")
            print ("\tTime cost = %.3f sec" % (e-s))
        
    def minibatch(self, batch_size):
        
        '''
        Minibatch method
        
        Extract a minibatch from current epoch for training. If the current 
        epoch is not enough, generate a new epoch and extract the minibatch
        from the new epoch.
        
        Parameters:
            
            batch_size: size of the minibatch
            
        Returns:
            
            A list of [sequences, labels, lengths]
            Sequences: a numpy.2darray with the shape of (bs, max_len). Each
                slice is a sequence, each element in a sequence is the index
                the token locate in the dictionary.
            Labels: a numpy.1darray with the shape of (bs). Each slice is an
                integer as the label index.
            Lengths: a numpy.1darray with the shape of (bs). Each slice is an
                integer as the length of the sequence.
        '''
        
        if self.__debug:
            s = time.time()
        
        # check if the batch size is larger than the size of the train-set
        if batch_size > len(self.__train["label"]):
            batch_size = len(self.__train["label"])
            if self.__debug:
                print ("Batch size larger than train#!")
        # check if a new epoch is needed
        if batch_size > len(self.__av):
            self.__av = random.sample(range(len(self.__train["label"])),
                                      len(self.__train["label"]))
            if self.__debug:
                print ("New epoch generated!")
            
        # get the index list
        chosen_idx = self.__av[:batch_size]
        self.__av = self.__av[batch_size:]
        batch = [[], [], []]

        for idx in chosen_idx:
            # get length
            batch[2].append(len(self.__train["seq"][idx]))
            # get label
            batch[1].append(self.__train["label"][idx]-1)   # 1-index to 0-index
            # get sequence
            tmp_seq = [0 for i in range(self.__max_len)]
            for seq_idx in range(len(self.__train["seq"][idx])):
                tmp_seq[seq_idx] = self.__dictionary.get(self.__train["seq"][idx][seq_idx],
                                                         self.__dict_size)
            batch[0].append(tmp_seq)
        
        batch[0] = numpy.asarray(batch[0], dtype=numpy.int32)
        batch[1] = numpy.asarray(batch[1], dtype=numpy.float32)
        batch[2] = numpy.asarray(batch[2], dtype=numpy.int32)
        
        if self.__debug:
            e = time.time()
            print ("Minibatch generated!!")
            print ("\tBatch size = %d" % batch_size)
            print ("\tTime cost = %.3f sec" % (e-s))
        
        return batch
        
    def test_batch(self, batch_size):
        
        '''
        Test batch method
        
        Extract a batch from test-set. Similar to minibatch method, but there
        is no such thing as "epoch" here. Each time, generate a batch from 
        the test-set. This means that there is a chance that a data pair may
        show up in two near call of this method.
        
        Parameters:
            
            batch_size: size of the test batch.
        
        Returns:
            
            Same as minibatch method. (It's driving me crazy to write these
            comments... So I'll try my best to write as less as possible.)
        '''
        
        if self.__debug:
            s = time.time()
        
        # check if the batch size is larger than the size of the test-set
        if batch_size > len(self.__test["label"]):
            batch_size = len(self.__test["label"])
            if self.__debug:
                print ("Batch size larger than test#!")
        # randomly generate the index list
        chosen_idx = random.sample(range(len(self.__test["label"])), batch_size)
        batch = [[] ,[], []]
        
        for idx in chosen_idx:
            # get length
            batch[2].append(len(self.__test["seq"][idx]))
            # get label
            batch[1].append(self.__test["label"][idx]-1)   # 1-index to 0-index
            # get sequence
            tmp_seq = [0 for i in range(self.__max_len)]
            for seq_idx in range(len(self.__test["seq"][idx])):
                tmp_seq[seq_idx] = self.__dictionary.get(self.__test["seq"][idx][seq_idx],
                                                         self.__dict_size)
            batch[0].append(tmp_seq)
        
        batch[0] = numpy.asarray(batch[0], dtype=numpy.int32)
        batch[1] = numpy.asarray(batch[1], dtype=numpy.float32)
        batch[2] = numpy.asarray(batch[2], dtype=numpy.int32)
        
        if self.__debug:
            e = time.time()
            print ("Test batch generated!!")
            print ("\tBatch size = %d" % batch_size)
            print ("\tTime cost = %.3f sec" % (e-s))
        
        return batch
        
    def valid_batch(self, batch_size):
        
        '''
        Valid batch method
        
        Extract a batch from valid-set. Same as test batch method
        
        Parameters:
            
            batch_size: size of the test batch.
        
        Returns:
            
            Same as test batch method.
        '''
        
        if self.__debug:
            s = time.time()
        
        # check if the batch size is larger than the size of the valid-set
        if batch_size > len(self.__valid["label"]):
            batch_size = len(self.__valid["label"])
            if self.__debug:
                print ("Batch size larger than valid#!")
        # randomly generate the index list
        chosen_idx = random.sample(range(len(self.__valid["label"])), batch_size)
        batch = [[] ,[], []]
        
        for idx in chosen_idx:
            # get length
            batch[2].append(len(self.__valid["seq"][idx]))
            # get label
            batch[1].append(self.__valid["label"][idx]-1)   # 1-index to 0-index
            # get sequence
            tmp_seq = [0 for i in range(self.__max_len)]
            for seq_idx in range(len(self.__valid["seq"][idx])):
                tmp_seq[seq_idx] = self.__dictionary.get(self.__valid["seq"][idx][seq_idx],
                                                         self.__dict_size)
            batch[0].append(tmp_seq)
        
        batch[0] = numpy.asarray(batch[0], dtype=numpy.int32)
        batch[1] = numpy.asarray(batch[1], dtype=numpy.float32)
        batch[2] = numpy.asarray(batch[2], dtype=numpy.int32)
        
        if self.__debug:
            e = time.time()
            print ("Valid batch generated!!")
            print ("\tBatch size = %d" % batch_size)
            print ("\tTime cost = %.3f sec" % (e-s))
        
        return batch

    def batch2raw(self, batch):
        
        '''
        Batch to raw data pair convertion method
        
        Convert a batch of tensors to (sequence, label) form.
        If an embedding is in the dictionary, we can recover the word token;
        However, if it's not in the dictionary (denoteded as "__UNKNOWN__"),
        it's of course impossible to recover it, so we simply put an
        "__UNKNOWN__" in the sequence.
        
        Parameters:
            
            batch: a batch generated by minibatch/test_batch/valid_batch
                method. It's a list of
                [sequence 2darray, label 1darray, length 1darray].
                
        Returns:
            
            A list of [question strings, label strings]
            The question strings are recovered from the sequence 3darray,
                and may contain "__UNKNOWN__" token.
            The label strings are recovered from the label 1darray, using
                the label dictionary.
        '''
        
        raw = [[], []]

        for s in range(len(batch[0])):
            # recover question string
            tmp_seq = ""
            for token in range(batch[2][s]):
                # look up the embedding in reverse dictionary
                tmp_seq += self.__rev_dictionary.get(batch[0][s][token],
                                                     "__UNKNOWN__") + " "
            raw[0].append(tmp_seq)
            
        for l in batch[1]:
            # recover label string
            # pay attention that in the raw data, labels are in 1-index form
            # but in the model, they are in 0-index form. So +1 is necessary.
            raw[1].append(self.__label_dictionary[l+1]) # 0-index back to 1-index
        
        return raw
        
    def check_dict(self, token):
        
        '''
        Check dictionary method
        
        Look up a token in the dictionary. If it's in the dictionary, return
        the index of the coresponding one-hot vector. Otherwise, it's not in
        the dictionary, return "UNKNOWN" then.
        
        Parameters:
            
            token: the token you want, which is a string.
            
        Returns:
            
            An integer if the token is in the dictionary, or a string
                "UNKNOWN" if it's not.
        '''
        
        return self.__dictionary.get(token, "UNKNOWN")
        
    def check_rev_dict(self, em_idx):
        
        '''
        Check reverse dictionary method
        
        Look up an one-hot vector in the reverse dictionary. If it's in the
        dictionary, return the coresponding token string. Otherwise, it's not
        in the dictionary, return "UNKNOWN" then.
        
        Parameters:
            
            em: the one-hot vector you want.
            
        Returns:
            
            A token string if the token is in the dictionary, or a string
                "UNKNOWN" if it's not.
        '''
        
        return self.__rev_dictionary.get(em_idx, "UNKNOWN")
        
    def check_label_dict(self, label):
        
        '''
        Check label method
        
        Look up a label in the label dictionary. 
        
        Parameters:
            
            label: the label you want, which is a integer (0-index).
            
        Returns:
            
            A string of the label.
        '''
        
        return self.__label_dictionary[label+1] # 0-index back to 1-index
        
    def get_max_seq_len(self):
        
        '''
        Get max sequence length method
        
        Get the max sequence length, which is defined during initilization.
        You cannot change it once defined. If you want to change, new another
        class please.
        The max length is used when generating a batch. We must pad all short
        sequences to the max length to generate a numpy.3darray.
        
        Parameters:
            
            None
            
        Returns:
            
            An integer as the max sequence length.
        '''
        
        return self.__max_len
    
    def get_vocab_size(self):
        
        '''
        Get vocabulary size method
        
        Get the size of the dictionary, which represents how many words the
        model knows. Of course the model knows much less than the vocabulary
        size of the dataset.
        The vocabulary size is used when generating a batch. The dimension of
        each token is vocab size + 1 (the 1 is "UNKNOWN").
        
        Parameters:
            
            None
            
        Returns:
            
            An integer as the vocabulary/dictionary size.
        '''
        
        return self.__dict_size
        
    def reset_rand_seed(self):
        
        '''
        Reset random seed method
        
        We use random and numpy package to generate batches. So if you change
        the random seed of these two packages during your use, please use this
        method to reset the random seed before you use anything in this class.
        Once called, the random seed is set to the rand_seed parameter you set
        during initialization (the default rand_seed is 1234).
        
        Parameters:
            
            None
            
        Returns:
            
            None
        '''
        
        random.seed(self.__rand_seed)
        numpy.random.seed(self.__rand_seed)
        
if __name__ == "__main__":
    
    ya = Yahoo_Answers_Dataset(debug=True)
    
    b_tr = ya.minibatch(32)
    b_te = ya.test_batch(5000)
    b_va = ya.valid_batch(1000)