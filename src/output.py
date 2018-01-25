
# coding: utf-8

# In[2]:

from gensim.models import Word2Vec # load wordEmbedding model
import numpy as np
import jieba, json, sys, os
os_path = os.getcwd()

################################################

input_path = sys.argv[1]
output_path = sys.argv[2]

# input_path = '.././'+input_path
# output_path = '.././'+output_path

################################################

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

def getTest_feature(test_data):
    paragraphs = []
    questions = []
    test_ids = []
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
                test_ids.append(qa['id']) # append question:string to tmp_x_row (behind context:string)

                #check if every question have unique answer
    return paragraphs, questions, test_ids

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


# load test json
test_data = load_json_data(input_path)

# retrieve data 
paragraphs, questions, test_ids = getTest_feature(test_data)


# get ans idxs
ans_list = []
cut_SE = np.load('prediction.npy').astype('int')
real_SE = map_Real_Index(cut_SE, paragraphs)

for i,SE in enumerate(real_SE):
    words = []
    for j in range(SE[0],SE[1]+1):
        words.append(str(j))
    ans_list.append(' '.join(words))

# output
o = open(output_path, 'w')
o.write("ID,Ans\n")
for qid, ans in zip(test_ids, ans_list):
    o.write("{},{}\n".format(qid, ans))
o.close()

