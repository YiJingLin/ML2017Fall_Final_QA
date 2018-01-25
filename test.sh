#!/bin/bash 
# $1 : testing data file path
# $2 : output csv file path
wget 'https://www.dropbox.com/s/kh9e6ply18lzofe/all_model_add_UNK100vec.bin?dl=1' -O'all_model_add_UNK100vec.bin'
wget 'https://www.dropbox.com/s/weogv9h3131iq74/all_model_add_UNK100vec.bin.syn1neg.npy?dl=1' -O'all_model_add_UNK100vec.bin.syn1neg.npy'
wget 'https://www.dropbox.com/s/k1zqofu7dnen2s7/all_model_add_UNK100vec.bin.wv.syn0.npy?dl=1' -O'all_model_add_UNK100vec.bin.wv.syn0.npy'
wget 'https://www.dropbox.com/s/v1bum9n1ybvdkn3/glove_vec.txt?dl=1' -O'glove_vec.txt'

# Handle worker mode
# mv all_model_add_UNK100vec.bin ./src
# mv all_model_add_UNK100vec.bin.syn1neg.npy ./src
# mv all_model_add_UNK100vec.bin.wv.syn0.npy ./src
# mv glove_vec.txt ./src
# mv model_weight.h5 ./src
# mv dict.txt.big ./src
# sleep 0.01 # wait handling complete

# main process
# python ./src/loadTest_predict.py $1
# sleep 0.05 # sleep 0.05 sec, wait prediction.npy complete
# python ./src/output.py $1 $2
# sleep 0.01 # wait process complete

# Handle worker mode
# mv src/all_model_add_UNK100vec.bin ./
# mv src/all_model_add_UNK100vec.bin.syn1neg.npy ./
# mv src/all_model_add_UNK100vec.bin.wv.syn0.npy ./
# mv src/glove_vec.txt ./
# mv src/model_weight.h5 ./
# mv src/dict.txt.big ./


python ./src/loadTest.py $1
sleep .1
python ./src/prediction.py
sleep .1
python ./src/output.py $1 $2