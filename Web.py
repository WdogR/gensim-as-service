import json
import time

import gensim
import numpy as np
from flask import Flask, request
from gensim.models import KeyedVectors

# json encoder
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
    
# embedding init
model_name = ['Baidu', 'Financial',  'Renmin', 'Sogou', 'Wiki', 'Zhihu']
# model_name = ['Baidu']
sample = True

app = Flask(__name__)
time0 = time.time()
model_list = []
vocab_list = []

if sample:
    for i in model_name:
        model_list.append(KeyedVectors.load_word2vec_format('embedding/sgns.' + i.lower() + '.word' + '.sample', binary=False))
else:
    for i in model_name:
        model_list.append(KeyedVectors.load_word2vec_format('embedding/sgns.' + i.lower() + '.word', binary=False))

for model in model_list:
    vocab_list.append(list(model.vocab.keys()))

time1 = time.time()

# Gensim Api
def single_embedding(word, model_index):
    model = model_list[model_index]
    vocab = vocab_list[model_index]
    if word in vocab:
        return 0, model.wv[word]
    else:
        return 1, None

def single_similarity(word1, word2, model_index):
    model = model_list[model_index]
    vocab = vocab_list[model_index]
    if word1 in vocab and word2 in vocab:
        return 0, model.wv.similarity(word1, word2)
    else:
        return 1, None

def single_topn_similar(n, model_index, pos_wordlist, neg_wordlist):
    model = model_list[model_index]
    vocab = vocab_list[model_index]
    tag = 0
    error_list = []
    if pos_wordlist:
        for i in pos_wordlist:
            if i not in vocab:
                error_list.append(i)
                tag = 1
    if neg_wordlist:
        for j in neg_wordlist:
            if j not in vocab:
                error_list.append(j)
                tag = 1
    if tag == 0:
        return tag, model.wv.most_similar(positive=pos_wordlist, negative=neg_wordlist, topn=n)
    else:
        return tag, error_list

def single_n_similarity(model_index, wordlist1, wordlist2):
    model = model_list[model_index]
    vocab = vocab_list[model_index]
    tag = 0
    error_list = []
    if wordlist1:
        for i in wordlist1:
            if i not in vocab:
                error_list.append(i)
                tag = 1
    if wordlist2:
        for j in wordlist2:
            if j not in vocab:
                error_list.append(j)
                tag = 1
    if tag == 0:
        return tag, model.wv.n_similarity(wordlist1, wordlist2)
    else:
        return tag, error_list

# Web Api
@app.route('/')
def hello():
    intro_dict = {'Loading_Time': time1-time0}
    for i in range(len(model_name)):
        intro_dict[model_name[i]+'_index']=i
    for i in range(len(model_name)):
        intro_dict[model_name[i]+'_size']=len(vocab_list[i])
    return json.dumps(intro_dict)


@app.route('/api/embedding', methods=['GET'])
def get_embedding():
    if request.method == 'GET':
        try:
            word = request.args.get('word')
            # print(word)
            final_result = {}
            for index in range(len(model_name)):
                tag, embedding = single_embedding(word, index)
                final_result[model_name[index]] = {'tag': tag ,'embedding': embedding}
                if tag == 1:
                    final_result[model_name[index]]['message'] = 'Not in the ' + model_name[index] + ' pre-trained model'
            return json.dumps({'code': 0, 'word': word, 'result': final_result}, cls=MyEncoder)
        except Exception as e:
            return json.dumps({'code': 1, 'message': str(e)})


@app.route('/api/similarity', methods=['GET'])
def get_similarity():
    if request.method == 'GET':
        try:
            word1 = request.args.get('word1')
            word2 = request.args.get('word2')
            # print(word1, word2)
            final_result = {}
            for index in range(len(model_name)):
                tag, similarity = single_similarity(word1, word2, index)
                final_result[model_name[index]] = {'tag': tag, 'similarity': similarity}
                if tag == 1:
                    final_result[model_name[index]]['message'] = 'Words not both in the ' + model_name[index] + ' pre-trained model'
            return json.dumps({'code': 0, 'word1': word1, 'word2': word2, 'result': final_result}, cls=MyEncoder)
        except Exception as e:
            return json.dumps({'code': 1, 'message': str(e)})


@app.route('/api/topn', methods=['GET'])
def get_topn():
    if request.method == 'GET':
        try:
            n = int(request.args.get('n'))
            poswordlist = request.args.get('pos')
            negwordlist = request.args.get('neg')
            if poswordlist is not None:
                poswordlist = poswordlist.strip().split(' ')
            if negwordlist is not None:
                negwordlist = negwordlist.strip().split(' ')

            final_result = {}

            for index in range(len(model_name)):
                tag, topn_or_error = single_topn_similar(n=n, model_index=index, pos_wordlist=poswordlist, neg_wordlist=negwordlist)
                final_result[model_name[index]] = {'tag': tag}
                if tag == 0:
                    final_result[model_name[index]]['top'+str(n)] = topn_or_error
                else:
                    error_string = ','.join(topn_or_error)
                    final_result[model_name[index]]['message'] = error_string + ' Not in ' + model_name[index] + ' pre-trained model'
            return json.dumps({'code': 0, 'pos_wordlist': poswordlist, 'neg_wordlist': negwordlist, 'result': final_result}, cls=MyEncoder)    
        except Exception as e:
            return json.dumps({'code': 1, 'message': str(e)})


@app.route('/api/n_similarity', methods=['GET'])
def get_n_similarity():
    if request.method == 'GET':
        try:
            wordlist1 = request.args.get('wlist1')
            wordlist2 = request.args.get('wlist2')
            if wordlist1 is not None:
                wordlist1 = wordlist1.strip().split(' ')
            if wordlist2 is not None:
                wordlist2 = wordlist2.strip().split(' ')

            final_result = {}

            for index in range(len(model_name)):
                tag, similarity_or_error = single_n_similarity(model_index=index, wordlist1=wordlist1, wordlist2=wordlist2)
                final_result[model_name[index]] = {'tag': tag}
                if tag == 0:
                    final_result[model_name[index]]['n_similarity'] = similarity_or_error
                else:
                    error_string = ','.join(similarity_or_error)
                    final_result[model_name[index]]['message'] = error_string + ' Not in ' + model_name[index] + ' pre-trained model'
            return json.dumps({'code': 0, 'wordlist1': wordlist1, 'wordlist2': wordlist2, 'result': final_result}, cls=MyEncoder)
        except Exception as e:
            return json.dumps({'code': 1, 'message': str(e)})


if __name__ == '__main__':
    app.run(host='10.108.17.226', port=8080, debug=True, threaded=True)