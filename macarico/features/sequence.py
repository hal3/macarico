from __future__ import division, generators, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import macarico
import macarico.util as util
from macarico.util import Var, Varng

class EmbeddingFeatures(macarico.StaticFeatures):
    def __init__(self,
                 n_types,
                 input_field = 'tokens',
                 d_emb = 50,
                 initial_embeddings = None,
                 learn_embeddings = True):

        d_emb = d_emb or initial_embeddings.shape[1]
        macarico.StaticFeatures.__init__(self, d_emb)

        self.input_field = input_field
        self.learn_embeddings = learn_embeddings
        self.embed_w = nn.Embedding(n_types, d_emb)
        
        if initial_embeddings is not None:
            e0_v, e0_d = initial_embeddings.shape
            assert e0_v == n_types, \
                'got initial_embeddings with first dim=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, \
                'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            self.embed_w.weight.data.copy_(torch.from_numpy(initial_embeddings))

        if not learn_embeddings:
            assert initial_embeddings is not None
            self.embed_w.requires_grad = False

    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        return self.embed_w(Varng(util.longtensor(self.embed_w.weight, txt))).view(1, env.N, self.dim)
    
    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        x = util.longtensor(self.embed_w.weight, batch_size, max_len).zero_()
        for n, txt in enumerate(txts):
            for i in range(txt_len[n]): # TODO could this be faster?
                x[n,i] = int(txt[i])
        return self.embed_w(Varng(x)).view(batch_size, max_len, self.dim) # TODO do we need to unpad somewhere?, txt_len

class BOWFeatures(macarico.StaticFeatures):
    def __init__(self,
                 n_types,
                 input_field='tokens',
                 window_size=0,
                 hashing=False):
        dim = (1 + 2 * window_size) * n_types
        macarico.StaticFeatures.__init__(self, dim)
        
        self.n_types = n_types
        self.input_field = input_field
        self.window_size = window_size
        self.hashing = hashing
        self._t = nn.Linear(1,1,bias=False)

    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        bow = util.zeros(self._t.weight, 1, len(txt), self.dim)
        self.set_bow(bow, 0, txt)
        return Varng(bow)

    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        bow = util.zeros(self._t.weight, batch_size, max_len, self.dim)
        for j, txt in enumerate(txts):
            self.set_bow(bow, j, txt)
        return Varng(bow)

    def set_bow(self, bow, j, txt):
        for n, word in enumerate(txt):
            for i in range(-self.window_size, self.window_size+1):
                m = n + i
                if m < 0: continue
                if m >= len(txt): continue
                v = (i + self.window_size) * self.n_types + self.hashit(word)
                bow[j,m,v] = 1
    
    def hashit(self, word):
        if self.hashing:
            word = (word + 48193471) * 849103817
        return int(word % self.n_types)

    # TODO _forward_batch

class RNN(macarico.StaticFeatures):
    def __init__(self,
                 features,
                 d_rnn = 50,
                 bidirectional = True,
                 n_layers = 1,
                 rnn_type = 'LSTM', # LSTM, GRU or RNN
                 ):
        # model is:
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - dimensionality
        #   n_layers  - how many layers of RNN
        #   bidirectional - is the RNN bidirectional?
        #   rnn_type - RNN/GRU/LSTM?
        # we assume that state:Env defines state.N and state.{input_field}
        macarico.StaticFeatures.__init__(self, d_rnn * (2 if bidirectional else 1))

        self.features = features
        self.bidirectional = bidirectional
        self.d_emb = features.dim
        self.d_rnn = d_rnn
        
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        self.rnn = getattr(nn, rnn_type)(self.d_emb,
                                         self.d_rnn,
                                         num_layers=n_layers,
                                         bidirectional=bidirectional,
                                         batch_first=True)

    def _forward(self, env):
        e = self.features(env)
        [res, _] = self.rnn(e)
        return res

    def _forward_batch(self, envs):
        e = self.features.forward_batch(envs)
        #if e is None: return None # someone below me doesn't support _forward_batch
        [res, _] = self.rnn(e) # TODO some stuff is padded, don't want to run backward LSTM on it!
        return res

class DilatedCNN(macarico.StaticFeatures):
    "see https://arxiv.org/abs/1702.02098"
    def __init__(self,
                 features,
                 n_layers=4,
                 passthrough=True,
                ):
        macarico.StaticFeatures.__init__(self, features.dim)
        self.features = features
        self.passthrough = passthrough
        self.conv = nn.ModuleList([nn.Linear(3 * self.dim, self.dim) \
                                   for i in range(n_layers)])
        self.oob = Parameter(torch.Tensor(self.dim))

    def _forward(self, env):
        l1, l2 = (0.5, 0.5) if self.passthrough else (0, 1)
        X = self.features(env).squeeze(0)
        N = X.shape[0]
        get = lambda XX, n: self.oob if n < 0 or n >= N else XX[n]
        dilation = [2 ** n for n in range(len(self.conv)-1)] + [1]
        for delta, lin in zip(dilation, self.conv):
            X = [l1 * get(X, n) + \
                 l2 * F.relu(lin(torch.cat([get(X, n),
                                            get(X, n-delta),
                                            get(X, n+delta)], dim=0))) \
                 for n in range(N)]
        return torch.cat(X, 0).view(1, N, self.dim)
            
    
class AverageAttention(macarico.Attention):
    arity = None # boil everything down to one item

    def __init__(self, features):
        macarico.Attention.__init__(self, features)
    
    def __call__(self, state):
        x = self.features(state)
        return [x.mean(dim=1)]

class AttendAt(macarico.Attention):
    """Attend to the current token's *input* embedding.
    """
    arity = 1
    def __init__(self, features, position='n'):
        macarico.Attention.__init__(self, features)
        self.position = (lambda state: getattr(state, position)) if isinstance(position, str) else \
                        position if hasattr(position, '__call__') else \
                        None
        assert self.position is not None
        self.oob = self.make_out_of_bounds()
        
    def forward(self, state):
        x = self.features(state)
        n = self.position(state)
        if n < 0: return [self.oob]
        if n >= len(x): return [self.oob]
        return [x[0,n].unsqueeze(0)]

    
class FrontBackAttention(macarico.Attention):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __init__(self, features):
        macarico.Attention.__init__(self, features)

    def forward(self, state):
        x = self.features(state)
        return [x[0,0].unsqueeze(0), x[0,-1].unsqueeze(0)]

class SoftmaxAttention(macarico.Attention):
    arity = None  # attention everywhere!
    actor_dependent = True
    
    def __init__(self, features, bilinear=True):
        macarico.Attention.__init__(self, features)
        self.bilinear = bilinear
        self.actor = [None] # put it in a list to hide it from pytorch? hacky???

    def set_actor(self, actor):
        assert self.actor[0] is None
        self.actor[0] = actor
        self.attention = nn.Bilinear(self.actor[0].dim, self.features.dim, 1) if self.bilinear else \
                         nn.Linear(self.actor[0].dim + self.features.dim, 1)

    def forward(self, state):
        x = self.features(state).squeeze(0)
        h = self.actor[0].hidden()
        N = x.shape[0]
        alpha = self.attention(h.repeat(N,1), x) if self.bilinear else \
                self.attention(torch.cat([h.repeat(N,1), x], 1))
        return [F.softmax(alpha.view(1,N), dim=1).mm(x)]
