from __future__ import division, generators, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.parameter import Parameter
import macarico

def getattr_deep(obj, field):
    for f in field.split('.'):
        obj = getattr(obj, f)
    return obj

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
                'got initial_embeddings with first dimp=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, \
                'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            self.embed_w.weight.data.copy_(torch.from_numpy(initial_embeddings))

        if not learn_embeddings:
            assert initial_embeddings is not None
            self.embed_w.requires_grad = False

    def _forward(self, state):
        txt = getattr_deep(state, self.input_field)
        return self.embed_w(Var(torch.LongTensor(txt), requires_grad=False)).view(state.N,1,-1)


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

    def _forward(self, state):
        txt = getattr_deep(state, self.input_field)
        bow = torch.zeros(len(txt), 1, self.dim)
        for n, word in enumerate(txt):
            for i in range(-self.window_size, self.window_size+1):
                m = n + i
                if m < 0: continue
                if m >= len(txt): continue
                v = (i + self.window_size) * self.n_types + self.hashit(word)
                bow[m,0,v] = 1

        return Var(bow, requires_grad=False)

    def hashit(self, word):
        if self.hashing:
            word = (word + 48193471) * 849103817
        return word % self.n_types
    
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
                                         num_layers = n_layers,
                                         bidirectional = bidirectional)

    def _forward(self, state):
        e = self.features(state)
        [res, _] = self.rnn(e)
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
        self.oob = Parameter(torch.Tensor(1, self.dim))

    def _forward(self, state):
        l1, l2 = (0.5, 0.5) if self.passthrough else (0, 1)
        X = self.features(state)
        N = X.shape[0]
        get = lambda XX, n: self.oob if n < 0 or n >= N else XX[n]
        dilation = [2 ** n for n in range(len(self.conv)-1)] + [1]
        for delta, lin in zip(dilation, self.conv):
            X = [l1 * get(X, n) + \
                 l2 * F.relu(lin(torch.cat([get(X, n),
                                            get(X, n-delta),
                                            get(X, n+delta)], 1))) \
                 for n in range(N)]
        return torch.cat(X, 0).view(N, 1, self.dim)
            
    
class AverageAttention(macarico.Attention):
    arity = None # boil everything down to one item

    def __init__(self, features):
        macarico.Attention.__init__(self, features)
    
    def __call__(self, state):
        x = self.features(state)
        return [x.mean(dim=0)]

class AttendAt(macarico.Attention):
    """Attend to the current token's *input* embedding.

    TODO: We should be able to attend to the *output* embeddings too, i.e.,
    embedding of the previous actions and hidden states.

    TODO: Will need to cover boundary token embeddings in some reasonable way.

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
        return [x[n]]

    
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
        return [x[0], x[-1]]

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
        x = self.features(state).squeeze(1)
        h = self.actor[0].hidden()
        N = x.shape[0]
        alpha = self.attention(h.repeat(N,1), x) if self.bilinear else \
                self.attention(torch.cat([h.repeat(N,1), x], 1))
        return [F.softmax(alpha.view(1,N), dim=1).mm(x)]
