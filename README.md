# Question Answering with Context in Traditional Chinese
> Department of Electronic Enginering, National Taiwan University \
> [Machine Learning (2017, Fall)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html) - Final Project
- kaggle competition ([link](https://www.kaggle.com/c/ml-2017fall-final-chinese-qa/leaderboard))

## Dependancy :
[![python version](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![keras version](https://img.shields.io/badge/Keras-2.0.8-green.svg)](https://pypi.python.org/pypi/Keras/2.0.8)
[![gensim version](https://img.shields.io/badge/gensim-3.0.1-green.svg)](https://pypi.python.org/pypi/gensim/3.0.1)
[![jieba version](https://img.shields.io/badge/jieba-0.39-green.svg)](https://pypi.python.org/pypi/jieba/)

## Introduction :
### Abstract :
受到 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 上 **[BiDAF + Self Attention](https://arxiv.org/abs/1611.01603)** 的啟發，我們採用 **self-attention** 以及 **Bidirectional Attention Flow** 的技術實作 QA model。

### Training :
採用既定json格式作為training data的input (請參照[train-v1.1.json](./data/train-v1.1.json))。在過程中，我們將 1.文章 2.問題 及 3.答案 抓出，分別存成3個list。文章及問題的list會經過word2vector將字詞轉換成向量，最後將3個list都轉成numpy array。文章及問題array作為model input，答案array作為model output。


```bash
python train.py <train.json>
```

### Testing :
一樣採用既定json格式作為testing data的input (請參照[test-v1.1.json](./data/test-v1.1.json))。與traing data process相比，我們只抓出文章及對應的問題，並經過word2vector轉換，最後將文章及問題arrays作為model input，並得到predict結果。

```bash
python test.py <test.json> <output.csv>
```

## Collaborators : 
### Team :
**NTU_R04942032_伊恩好夥伴** \
(inspired from Ian Goodfellow)

### member :
- Huang.Ychen (黃彥澄) ([github](#))
- Lin.ZhuangYing (林宗穎) ([github](#))
- Lin.YiJing (林益璟) ([github](#))

