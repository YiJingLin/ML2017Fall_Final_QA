import json, jieba, sys, math
jieba.set_dictionary('./data/dict.txt.big') # freindly to traditional chinese


class Load:
	def __init__(self):
		return

	def loadTestData(self, path):
		test_data = self._load_json_data(path)
		return self._getTest_feature(test_data)

	def loadTrainData(self, path):
		train_data = self._load_json_data(path) 		
		return self._getTrain_feature_label(train_data)

	def loadTestID(self, path):
	    test_data = self._load_json_data(path)

	    test_id = [] # [<context>,<quuetion>] for each row
	    
	    #get data position 
	    subjects = test_data['data']

	    for subject in subjects: 
	        # subject contains title and *paragraphs*
	        for paragraph in subject['paragraphs']:
	            # paragraphs contains *context* and *qas*	            
	            for qa in paragraph['qas']:
	                
	                ######################################
	                # row data in train_x
	                test_id.append(qa['id']) # append question:string to tmp_x_row (behind context:string)
	                #######################################
	                #check if every question have unique answer
	    return test_id

	def _load_json_data(self, path):
	    data = []
	    
	    # load text
	    with open(path, 'r', encoding='utf-8') as file:
	        for line in file:
	            data.append(line)
	    
	    # trans to json
	    # default data.length=1, so only pick first element in data
	    data = json.loads(data[0]) 
	    return data

	def _getTrain_feature_label(self, train_data):
	    train_x = [] # [<context>,<quuetion>] for each row 
	    train_y = [] # [<start of answer>,<end of answer>] for each row
	    
	    # store each data row temporally
	    tmp_x_row = [] 
	    tmp_y_row = []
	    
	    #get data position 
	    subjects = train_data['data']

	    for subject in subjects: 
	        # subject contains title and *paragraphs*
	        
	        for paragraph in subject['paragraphs']:
	            # paragraphs contains *context* and *qas*
	            context = paragraph['context']#.replace("\n","")
	            
	            for qa in paragraph['qas']:
	                
	                ######################################
	                # row data in train_x
	                tmp_x_row=[context] # replace last data in tmp_x_row with context:string
	                tmp_x_row.append(qa['question']) # append question:string to tmp_x_row (behind context:string)
	                train_x.append(tmp_x_row)
	                # row data in train_y
	                answer = qa['answers'][0]
	                tmp_y_row = [answer['answer_start']]
	                tmp_y_row.append(answer['answer_start']+len(answer['text'])) # answer_end
	                #tmp_y_row.append(answer['text']) #add answer text for checking
	                train_y.append(tmp_y_row)
	                #######################################
	                #check if every question have unique answer
	                if not len(qa['answers']) ==1 :
	                    print('more than one answer!!!')                
	    
	    return train_x, train_y

	def _getTest_feature(self, test_data):
	    test_x = [] # [<context>,<quuetion>] for each row
	    
	    # store each data row temporally
	    tmp_x_row = [] 

	    #get data position 
	    subjects = test_data['data']
	    
	    for subject in subjects: 
	        # subject contains title and *paragraphs*
	        
	        for paragraph in subject['paragraphs']:
	            # paragraphs contains *context* and *qas*
	            context = paragraph['context']#.replace("\n","")
	            
	            for qa in paragraph['qas']:
	                
	                ######################################
	                # row data in train_x
	                tmp_x_row=[context] # replace last data in tmp_x_row with context:string
	                tmp_x_row.append(qa['question']) # append question:string to tmp_x_row (behind context:string)
	                test_x.append(tmp_x_row)
	                #######################################
	                #check if every question have unique answer
	    return test_x

class Process:
	def __init__(self, stop_list=None):
		self.stop_list=[]
		self.prog_bar_total = 0
		self.prog_bar_count = 0

		if not stop_list==None:
			self.stop_list = stop_list

	def jieba_tokenize_train_data(self, x):
		result_x = []
		tmp_row = []

		self.prog_bar_count=0
		self.prog_bar_total=len(x)

		for paragraph, question in x:

			tmp_row = []
			tmp_row.append(list(jieba.cut(paragraph, cut_all=False)))
			tmp_row.append(list(jieba.cut(question, cut_all=False)))
			result_x.append(tmp_row)

			self._update_progress_bar()
			self.prog_bar_count +=1
		
		return result_x

	def remove_Xs_mark_according_stop_list(self, train_x, train_y, stop_list=None):
		result_x = []
		result_y = []
		invalid_list = []

		if stop_list==None:
			stop_list = self.stop_list

		self.prog_bar_count = 0
		self.prog_bar_total = len(train_x)

		for idx, paragraph_question_list in enumerate(train_x):
			tmp_x_row = []
			
			### paragraph process
			paragraph_list = paragraph_question_list[0]
			bias = 0
			tmp_para_list = []
			tmp_origin_para_list=[] # store bias into original para_tuple_list

			for word in paragraph_list:

				if word in stop_list:
					bias +=1 # find new mark tuple, then add bias 
				else:
					tmp_para_list.append(word)

				new_tuple = (word, bias)
				tmp_origin_para_list.append(new_tuple)

			tmp_x_row.append(tmp_para_list)

			### ans process : adjust ans indx according to bias
			# shift ans by removed-marks
			tmp_y, valid = self._get_new_ans_idx_from_word_tuple_list(tmp_origin_para_list, train_y[idx]) 
			# shift ans by jieba paragraph
			tmp_y, valid = self._get_jieba_ans_idx_from_tmp_para_list(tmp_para_list, tmp_y)

			### question process
			question_list = paragraph_question_list[1]
			tmp_ques_list = []

			for word in question_list:
				if word not in stop_list:
					tmp_ques_list.append(word)

			tmp_x_row.append(tmp_ques_list)

			### finally : merge paragraph and question, then print progress bar (optional)
			if valid:
				result_y.append(tmp_y)
				result_x.append(tmp_x_row)
			else:
				invalid_list.append(idx)
			self._update_progress_bar()
			self.prog_bar_count +=1

		return result_x, result_y, invalid_list


	def remove_Ys_mark_according_stop_list(self, test_x, stop_list=None):
		result_x = []
		bias_list = []

		if stop_list==None:
			stop_list = self.stop_list

		self.prog_bar_count = 0
		self.prog_bar_total = len(test_x)

		for idx, paragraph_question_list in enumerate(test_x):
			tmp_x_row = []
			
			### paragraph process
			paragraph_list = paragraph_question_list[0]
			bias = 0
			tmp_para_list = []
			tmp_origin_para_list=[] # store bias into original para_tuple_list

			for word in paragraph_list:

				if word in stop_list:
					bias +=1 # find new mark tuple, then add bias 
				else:
					tmp_para_list.append(word)

				new_tuple = (word, bias)
				tmp_origin_para_list.append(new_tuple)

			tmp_x_row.append(tmp_para_list)
			
			### bias_list append
			bias_list.append(self._get_bias_list(tmp_origin_para_list))

			### question process
			question_list = paragraph_question_list[1]
			tmp_ques_list = []

			for word in question_list:
				if word not in stop_list:
					tmp_ques_list.append(word)

			tmp_x_row.append(tmp_ques_list)

			### finally : merge paragraph and question, then print progress bar (optional)
			result_x.append(tmp_x_row)

			self._update_progress_bar()
			self.prog_bar_count +=1

		return result_x, bias_list


	def _get_bias_list(self, tuple_list_withBias):
		bias_list = []
		for word, bias in tuple_list_withBias: #(word, bias)
			for times in range(len(word)):
				bias_list.append(bias)
		
		return bias_list
	def _get_new_ans_idx_from_word_tuple_list(self, tuple_list_withBias, ans):
		valid=True
		bias_list = []
		for word, bias in tuple_list_withBias: #(word, bias)
			for times in range(len(word)):
				bias_list.append(bias)

		original_ans_start = ans[0]
		original_ans_end = ans[1]
		
		new_ans_start = new_ans_end = 0

		try:
			new_ans_start = original_ans_start - bias_list[original_ans_start]
			new_ans_end = original_ans_end - bias_list[original_ans_end-1]
		except:
			print('invalid ans in mark-remove stage : data row no.' +str(self.prog_bar_count))
			valid=False
		return [new_ans_start, new_ans_end], valid

	def _get_jieba_ans_idx_from_tmp_para_list(self, para_list, ans):
		valid = True
		tmp_bias_list = []
		for bias, word in enumerate(para_list):
			for times in range(len(word)):
				tmp_bias_list.append(bias)
		try:
			ans[0] = tmp_bias_list[ans[0]]
			ans[1] = tmp_bias_list[ans[1]-1]
		except:
			print('invalid ans in jieba paragraph stage : data row no.' +str(self.prog_bar_count))
			valid = False
		return ans, valid

	def _jieba_tokenize(self, sentence):
		return jieba.tokenize(sentence,'utf-8')


	def progressbar(self, cur, total):
		cur +=1
		percent = '{:.2%}'.format(cur / total)
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %s" % (
	    						'=' * int(math.floor(cur * 50 / total)),
	    						percent))
		if cur == total:
			sys.stdout.write('\n')

		sys.stdout.flush()

	def _update_progress_bar(self):
		total = self.prog_bar_total
		cur = self.prog_bar_count
		cur +=1
		percent = '{:.2%}'.format(cur / total)
		sys.stdout.write('\r')
		sys.stdout.write("[%-50s] %s" % (
	    						'=' * int(math.floor(cur * 50 / total)),
	    						percent))
		if cur == total:
			sys.stdout.write('\n')
		sys.stdout.flush()
