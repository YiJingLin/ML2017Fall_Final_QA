
# coding: utf-8

# In[ ]:

from gensim.models import Word2Vec # load wordEmbedding model
import numpy as np
import jieba, json, sys, math, csv

#freindly to traditional chinese
jieba.set_dictionary('dict.txt.big')
#######################################################

input_path = sys.argv[1]
# input_path = '.././'+input_path
#######################################################

# In[ ]:

def load_json_data(path):
    data = []

    # load text
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line)

    # trans to json
    # default data.length=1, so only pick first element in data
    data = json.loads(data[0]) 
    return data

test_data = load_json_data(input_path)


# In[ ]:

def getTest_feature(test_data):
    paragraphs = []
    questions = []

    # store each data row temporally
    tmp_x_row = [] 

    #get data position 
    subjects = test_data['data']

    for subject in subjects: 
        # subject contains title and *paragraphs*

        for paragraph in subject['paragraphs']:
            # paragraphs contains *context* and *qas*
            context = list(jieba.tokenize(paragraph['context'].replace("\n","")))

            for qa in paragraph['qas']:

                ######################################
                paragraphs.append(context)
                questions.append(list(jieba.tokenize(qa['question'])))
                #######################################
                #check if every question have unique answer
    return paragraphs, questions


paragraphs, questions = getTest_feature(test_data)

del jieba
# In[ ]:

import pickle
f = open('glove_vec.txt','rb')
glove_dict = pickle.load(f)
f.close()

from gensim.models import Word2Vec
wv = Word2Vec.load('all_model_add_UNK100vec.bin')


test_paragraph = np.zeros((len(paragraphs),650,200))
test_question = np.zeros((len(paragraphs),60,200))

for i,s in enumerate(paragraphs) :
    for j,w in enumerate(s) :
        if not w[0] in wv.wv :
            test_paragraph[i,j,:100] = wv.wv["<UNK>"]
            test_paragraph[i,j,100:] = np.array(glove_dict["<UNK>"]).astype('float32')
        else:
            test_paragraph[i,j,:100] = wv.wv[w[0]]
            test_paragraph[i,j,100:] = np.array(glove_dict[w[0]]).astype('float32')
for i,s in enumerate(questions) :
    for j,w in enumerate(s) :
        if not w[0] in wv.wv :
            test_question[i,j,:100] = wv.wv["<UNK>"]
            test_question[i,j,100:] = np.array(glove_dict["<UNK>"]).astype('float32')
        else:
            test_question[i,j,:100] = wv.wv[w[0]]
            test_question[i,j,100:] = np.array(glove_dict[w[0]]).astype('float32')

np.save('test_paragraph_WVboth.npy',test_paragraph)
np.save('test_question_WVboth.npy',test_question)


print('succesfully load data.')



# In[ ]:




# In[ ]:



