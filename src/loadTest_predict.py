
# coding: utf-8

# In[ ]:

from gensim.models import Word2Vec # load wordEmbedding model
import numpy as np
import jieba, json, sys, math, csv

from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.layers.core import *
from keras.models import *
from keras.utils import plot_model
from keras.activations import softmax

PARAGRAPH_LENGTH = 650
QUESTION_LENGTH = 60
W2V_LENGTH = 200

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

def old_map_Real_Index(modelPreds, paragraphs): # include start and end index
    result = []
    if not len(modelPreds)==len(paragraphs):
        print("invalid input length, plz check these list data")
        return result # []
    row = []
    for idxPrd, pred in enumerate(modelPreds):
        row = [paragraphs[idxPrd][pred[0]][-2]] #contain start index and end index
        row.append(paragraphs[idxPrd][pred[1]][-1])
        result.append(row)
    return result


# In[ ]:

def map_Real_Index(modelPreds, paragraphs): # include start and end index
    result = []
    if not len(modelPreds)==len(paragraphs):
        print("invalid input length, plz check these list data")
        return result # []
    row = []
    for idxPrd, pred in enumerate(modelPreds):
        if pred[1] >= len(paragraphs[idxPrd]):
            pred[1] = len(paragraphs[idxPrd])-1
        if pred[0] >= len(paragraphs[idxPrd]):
            pred[0] = len(paragraphs[idxPrd])-1
        row = [paragraphs[idxPrd][pred[0]][1]] #contain start index and end index
        row.append(paragraphs[idxPrd][pred[1]][2])
        result.append(row)
    return result


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

# np.save('test_paragraph_WVboth.npy',test_paragraph)
# np.save('test_question_WVboth.npy',test_question)


# In[ ]:

def QA_model_AF(lstm_units=64):
    # Contextual Embedding Layer
    paragraph = Input(shape=(PARAGRAPH_LENGTH,W2V_LENGTH),name='INPUT_paragraph')
    question = Input(shape=(QUESTION_LENGTH,W2V_LENGTH),name='INPUT_question')
    
    q = Bidirectional(LSTM(lstm_units, return_sequences=True))(question)
    p = Bidirectional(LSTM(lstm_units, return_sequences=True))(paragraph)
    
    p_c = Flatten()(p)
    p_c = RepeatVector(QUESTION_LENGTH)(p_c)
    p_c = Reshape((QUESTION_LENGTH,PARAGRAPH_LENGTH,lstm_units*2))(p_c)
    p_c = Permute((2,1,3))(p_c)
    
    q_c = Flatten()(q)
    q_c = RepeatVector(PARAGRAPH_LENGTH)(q_c)
    q_c = Reshape((PARAGRAPH_LENGTH,QUESTION_LENGTH,lstm_units*2))(q_c)
    
    # Making Similarity Matrix
    m = Multiply()([p_c,q_c])
    s0 = Concatenate(axis=3)([p_c,q_c,m])
    s1 = Dense(units=1)(s0)
    s = Reshape((PARAGRAPH_LENGTH,QUESTION_LENGTH))(s1)
    
    # Attetion Flow
    c2q = Lambda(lambda x: softmax(x,axis=2))(s)
    c2q_c = Flatten()(c2q)
    c2q_c = RepeatVector(lstm_units*2)(c2q_c)
    c2q_c = Reshape((lstm_units*2,PARAGRAPH_LENGTH,QUESTION_LENGTH))(c2q_c)
    c2q_c = Permute((2,3,1))(c2q_c)
    m_q = Multiply()([c2q_c,q_c])
    q_att = Lambda(lambda x: K.sum(x,axis=2))(m_q)
    
    q2c = Lambda(lambda x: K.max(x,axis=2))(s)
    q2c = Lambda(lambda x: K.softmax(x))(q2c)
    q2c_c = RepeatVector(lstm_units*2)(q2c)
    q2c_c = Permute((2,1))(q2c_c)
    m_p = Multiply()([q2c_c,p])
    
    p_att = Lambda(lambda x: K.sum(x,axis=1))(m_p)
    p_att = RepeatVector(PARAGRAPH_LENGTH)(p_att)
    
    g_0 = Multiply()([p,q_att])
    g_1 = Multiply()([p,p_att])
    G = Concatenate(axis=2)([p,q_att,g_0,g_1])
    
    # Modeling Layer
    M_0 = Bidirectional(LSTM(lstm_units, return_sequences=True))(G)
    M_1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(M_0)
    M_2 = Bidirectional(LSTM(lstm_units, return_sequences=True))(M_1)
    
    # Output Layer
    concat_start = Concatenate(axis=2)([G,M_0]) # cut here
    concat_end = Concatenate(axis=2)([G,M_1]) # cut here
    
    p_start = Dense(1)(concat_start)
    p_end = Dense(1)(concat_end)
    p_start = Flatten()(p_start)
    p_end = Flatten()(p_end)
    
    start = Activation('softmax')(p_start)
    end = Activation('softmax')(p_end)
    
    model = Model(input=[paragraph,question], output=[start,end])
    
    return model


# In[ ]:

model = QA_model_AF(80)
model.load_weights('model_weight.h5')


# In[ ]:

# test_p = np.load('test_paragraph_WVboth.npy')
# test_q = np.load('test_question_WVboth.npy')

result = model.predict([test_paragraph,test_question])

predict_DPdecay = np.zeros((len(result[0]),2))
M_decay = np.ones((PARAGRAPH_LENGTH,PARAGRAPH_LENGTH))
for s in range(PARAGRAPH_LENGTH):
    for e in range(PARAGRAPH_LENGTH):
        if s>e :
            M_decay[s,e] = 0
        elif e-s>10 :
            M_decay[s,e] = 1/(1+0.01*(e-s-10)**2)

for i in range(len(result[0])):
    M = np.dot(result[0][i].reshape((-1,1)),result[1][i].reshape((1,-1)))
    M = np.multiply(M,M_decay)
    index = np.argmax(M)
    predict_DPdecay[i,0] = index//PARAGRAPH_LENGTH
    predict_DPdecay[i,1] = index%PARAGRAPH_LENGTH
    
np.save('prediction.npy',predict_DPdecay)
print('succesfully predict the answers.')



# In[ ]:




# In[ ]:



