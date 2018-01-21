# Question Answering with Context in Traditional Chinese
> Department of Electronic Enginering, National Taiwan University \
> [Machine Learning (2017, Fall)](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17_2.html) - Final Project
- kaggle competition ([link](https://www.kaggle.com/c/ml-2017fall-final-chinese-qa/leaderboard))
## 

[![pypi package](https://img.shields.io/pypi/v/python-symphony.svg)](https://pypi.python.org/pypi/python-symphony/)


## Introduction :
### Abstract :
受到 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 上 **[BiDAF + Self Attention](https://arxiv.org/abs/1611.01603)** 的啟發，我們採用 **self-attention** 以及 **Bidirectional Attention Flow** 的技術實作 QA model。

### Training :
採用既定json格式作為training data的input (請參照[train-v1.1.json](./data/train-v1.1.json))。在過程中，我們會先將 1.文章 2.問題 及 3.答案 抓出，並分別存成3個list。文章及問題的list會經過word2vector將字詞轉換成向量，

## Collaborators : 
### Team :
**NTU_R04942032_伊恩好夥伴** \
(inspired from Ian Goodfellow)

### member :
- Huang.Ychen (黃彥澄) ([github](#))
- Lin.ZhuangYing (林宗穎) ([github](#))
- Lin.YiJing (林益璟) ([github](#))

