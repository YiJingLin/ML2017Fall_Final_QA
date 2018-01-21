#!/bin/bash 
# $1 : testing data file path
# $2 : output csv file path
wget 'https://www.dropbox.com/s/kh9e6ply18lzofe/all_model_add_UNK100vec.bin?dl=1' -O'all_model_add_UNK100vec.bin'
wget 'https://www.dropbox.com/s/weogv9h3131iq74/all_model_add_UNK100vec.bin.syn1neg.npy?dl=1' -O'all_model_add_UNK100vec.bin.syn1neg.npy'
wget 'https://www.dropbox.com/s/k1zqofu7dnen2s7/all_model_add_UNK100vec.bin.wv.syn0.npy?dl=1' -O'all_model_add_UNK100vec.bin.wv.syn0.npy'
wget 'https://www.dropbox.com/s/v1bum9n1ybvdkn3/glove_vec.txt?dl=1' -O'glove_vec.txt'

python loadTest_predict.py $1
python output.py $1 $2
