import numpy as np
import math
import chainer
from chainer import Chain
from chainer import Variable
from chainer.functions import *
import chainer.links as L
import chainer.functions as F

#VOCAB_SIZE是单词的数量、HIDDEN_SIZE是隐藏层的维数
#读那个文章进来
text = open("text.txt").read().lower().split()
text_data = sorted(list(set(text)))
vocab_size = len(text_data)
HIDDEN_SIZE = 10

W = np.array(  [[0, 0, 0],
                [1, 1, 1],
                [2, 2, 2]]  )

class MyChain(Chain.links):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.w_xh=L.EmbedID(vocab_size, HIDDEN_SIZE)
            self.w_hh=L.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.w_hy=L.Linear(HIDDEN_SIZE, vocab_size)


    def __call__(self, x):
        h1 = self.w_xh(x)
        h2 = self.w_hh(F.sigmoid(h1))
        return self.w_hy(F.sigmoid(h2))

model = MyChain()
#model = L.Classifier(chain)
#model = MyChain()

def convert_to_your_word_id(word):
    return text_data.index(word)

def forward(sentence, model): # sentence是str的排列结果。
    sentence = [convert_to_your_word_id(word) for word in sentence] # 单词转换为ID
    h = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32)) # 隐藏层的初值
    accum_loss = Variable(np.zeros((), dtype=np.float32)) # 累计损失的初値
    log_joint_prob = float(0) # 句子的结合概率

    for word in sentence:
        x = Variable(np.array([[word]], dtype=np.int32)) # 下一次的输入层
        u = model.w_hy(h)
        accum_loss += softmax_cross_entropy(u, x) # 累计损失
        y = softmax(u) # 下一个单词的概率分布
        log_joint_prob += math.log(y.data[0][word]) #结合概率的分布
        h = tanh(model.w_xh(x) + model.w_hh(h)) #隐藏层的更新

    return log_joint_prob, accum_loss  #返回结合概率的计算结果


def train(sentence_set, model):
    opt = chainer.optimizers.MomentumSGD() # 使用梯度下降法
    opt.use_cleargrads()
    opt.setup(model) # 学习初期化
    for sentence in sentence_set:
        opt.zero_grad() # 勾配の初期化
        log_joint_prob, accum_loss = forward(sentence, model) # 损失的计算
        accum_loss.backward() # 误差反向传播
        opt.clip_grads(10) # 剔除过大的梯度
        opt.update() # 参数更新

# model = Chain(
#     w_xh = L.EmbedID(vocab_size, hidden_size), # 输入层(one-hot) -> 隐藏层
#     w_hh = L.Linear(hidden_size, hidden_size), # 隐藏层 -> 隐藏层
#     w_hy = L.Linear(hidden_size, vocab_size), # 隐藏层 -> 输出层
# )

train(model,text)