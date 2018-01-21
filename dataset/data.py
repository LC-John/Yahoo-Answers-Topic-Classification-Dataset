# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:27:20 2018

@author: DrLC
"""

import pandas
import math
import nltk
import re
import gzip, pickle
import matplotlib.pyplot as plt

def load_csv(path="test.csv"):
    
    return pandas.read_csv(path, header=None)
    
def load_class(path="classes.txt"):
    
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    
    ret = [line.strip('\n') for line in lines]
    return ret
    
def simple_clean_sentence(sent):
    
    punctuation = ['.', ',', '!', '/', ':', ';',
                   '+', '-', '*', '?', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '$', '#',
                   '"', '`', '^']

    sent = sent.replace('\n', ' ').replace('\\n', ' ').replace('\\', ' ')
    for p in punctuation:
        sent = sent.replace(p, ' '+p+' ')
    sent = re.sub(r'\d+\.?\d*', ' numbernumbernumbernumbernumber ', sent)
    return sent.lower()
    
def extract_questions_and_labels(d, lookup_table):
    
    ret = {"text":[], "label":[], "lookup_table":[]}
    for idx in range(len(d[0])):
        ret["label"].append(d[0][idx])
        appd = ""
        if type(d[2][idx]) is float and math.isnan(d[2][idx]):
            appd = ""
        else:
            appd = simple_clean_sentence(d[2][idx])
        ret["text"].append((simple_clean_sentence(d[1][idx])+' '+appd).lower())
    
    ret["lookup_table"] = lookup_table
    return ret
    
def word_tokenize(d, max_len=100):
    
    for idx in range(len(d['text'])):
        d['text'][idx] = nltk.word_tokenize(d['text'][idx])
        
    ret = {"text":[], "label":[], "lookup_table":[]}
    ret["lookup_table"] = d["lookup_table"]
    for idx in range(len(d["text"])):
        if len(d["text"][idx]) <= max_len:
            ret["text"].append(d["text"][idx])
            ret["label"].append(d["label"][idx])
    return ret
    
def save_dataset(d, path="yahoo_answers_test.pkl.gz"):
    
    f = gzip.open(path, "wb")
    pickle.dump(d, f)
    f.close()
    
def load_dataset(path='yahoo_answers_test.pkl.gz'):
    
    f = gzip.open(path, "rb")
    d = pickle.load(f)
    f.close()
    
    return d
    
def stat_word(d):
    
    stat = {}
    
    for s in d["text"]:
        for w in s:
            if w in stat.keys():
                stat[w] += 1
            else:
                stat[w] = 1
    stat = sorted(stat.items(), key=lambda d:d[1], reverse=True)    

    return stat

def stat_sentence_length(d):
    
    stat = {}

    for s in d["text"]:
        l = len(s)
        if l in stat.keys():
            stat[l] += 1
        else:
            stat[l] = 1
    stat = sorted(stat.items(), key=lambda d:d[0], reverse=True) 
    
    return stat
    
def stat_label(d):
    
    stat = {}

    for l in d["label"]:
        if l in stat.keys():
            stat[l] += 1
        else:
            stat[l] = 1
    
    return stat
    
def generate_pkl_gz(max_len=100,
                    class_path="classes.txt",
                    test_src_path="test.csv",
                    train_src_path="train.csv",
                    test_tgt_path="yahoo_answers_test.pkl.gz",
                    train_tgt_path="yahoo_answers_train.pkl.gz"):
    
    lt = load_class(path=class_path)
    print ("Class lookup table loaded!")
    
    d_tr_ = load_csv(path=train_src_path)
    print ("All training data loaded!")
    d_tr = extract_questions_and_labels(d_tr_, lt)
    print ("Training data extracted!")
    d_tr = word_tokenize(d_tr, max_len)
    print ("Training data word token generated!")
    save_dataset(d_tr, path=train_tgt_path)
    print ("Training data saved!")
    
    d_te_ = load_csv(path=test_src_path)
    print ("All test data loaded!")
    d_te = extract_questions_and_labels(d_te_, lt)
    print ("Test data extracted!")
    d_te = word_tokenize(d_te, max_len)
    print ("Test data word token generated!")
    save_dataset(d_te, path=test_tgt_path)
    print ("Test data saved!")
    
    return d_tr, d_te
    
def generate_stat_word(tr=None, te=None, d=None,
                       train_path="yahoo_answers_train.pkl.gz",
                       test_path="yahoo_answers_test.pkl.gz",
                       dict_path="yahoo_answers_dict.pkl.gz"):
    
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print ("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print ("Train set loaded!")
    
    if d is None:
        d = {"text":[], "label":[], "lookup_table":[]}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]
    
    s_word = stat_word(d)
    f = open("word_stat.txt", "w", encoding="UTF-8")
    for inst in s_word:
        f.write(str(inst[1])+"\t"+inst[0]+"\n")
    f.close()
    
    f = gzip.open(dict_path, "wb")
    pickle.dump(s_word, f)
    f.close()
    
    return s_word, d
    
def generate_stat_sentence_length(tr=None, te=None, d=None,
                                  train_path="yahoo_answers_train.pkl.gz",
                                  test_path="yahoo_answers_test.pkl.gz"):
    
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print ("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print ("Train set loaded!")
    
    if d is None:
        d = {"text":[], "label":[], "lookup_table":[]}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]
    
    s_senlen = stat_sentence_length(d)
    
    count = [i[1] for i in s_senlen]
    length = [i[0] for i in s_senlen]
    plt.plot(length, count, 'ro')
    plt.savefig("len_distribution.png")
    plt.show()

    return s_senlen, d
    
def generate_stat_label(tr=None, te=None, d=None,
                        train_path="yahoo_answers_train.pkl.gz",
                        test_path="yahoo_answers_test.pkl.gz"):
    
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print ("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print ("Train set loaded!")
    
    if d is None:
        d = {"text":[], "label":[], "lookup_table":[]}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]
    
    s_label = stat_label(d)
    s_label = s_label.items()
    
    count = [i[1] for i in s_label]
    length = [i[0] for i in s_label]
    plt.plot(length, count, 'ro')
    plt.savefig("label_distribution.png")
    plt.show()

    return s_label, d

if __name__ == "__main__":
    
    #d_tr, d_te = generate_pkl_gz(max_len=100)
    
    #s_word, d = generate_stat_word()
    
    s_senlen, d = generate_stat_sentence_length()
    
    s_label, d = generate_stat_label(d=d)