import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import macarico
import macarico.util as util
from macarico.util import Var, Varng

qrnn_available = False
try:
    from torchqrnn import QRNN
    qrnn_available = True
except ImportError:
    pass


class EmbeddingFeatures(macarico.StaticFeatures):
    def __init__(self,
                 n_types,
                 input_field = 'X',
                 d_emb = 50,
                 initial_embeddings = None,
                 learn_embeddings = True):

        d_emb = d_emb or initial_embeddings.shape[1]
        macarico.StaticFeatures.__init__(self, d_emb)

        self.input_field = input_field

        # setup: we always learn an _offset_ to the intially provided embeddings (akin to regularizing toward them).
        # what this means is:
        # if initial_embeddings are defined and learn_embeddings is False --> just use the initial_embeddings
        # if initial_embeddings are defined and learn_embeddings is True  --> learn new embeddings, added to old, initialized to zero
        # if initial_embeddings don't exist and learn_embeddings is False --> throw exception
        # if initial_embeddings don't exist and learn_embeddings is True  --> learn embeddings from scratch
        self.initial_embeddings = None
        self.learned_embeddings = None
        if initial_embeddings is None:
            assert learn_embeddings, 'must either be learning embeddings or have initial_embeddings != None'
            self.learned_embeddings = nn.Embedding(n_types, d_emb)
        else: # we have initial embeddings
            e0_v, e0_d = initial_embeddings.shape
            assert e0_v == n_types, 'got initial_embeddings with first dim=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, 'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            self.initial_embeddings = nn.Embedding(n_types, d_emb)
            self.initial_embeddings.weight.data.copy_(torch.from_numpy(initial_embeddings))
            self.initial_embeddings.requires_grad = False
            if learn_embeddings:
                self.learned_embeddings = nn.Embedding(n_types, d_emb)
                self.learned_embeddings.weight.data.zero_()

    def embed(self, txt_var):
        if self.initial_embeddings is None:
            return self.learned_embeddings(txt_var)
        if self.learned_embeddings is None:
            return self.initial_embeddings(txt_var)
        return self.learned_embeddings(txt_var) + self.initial_embeddings(txt_var)
                
    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        return self.embed(Varng(util.longtensor(self, txt))).view(1, -1, self.dim)
    
    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        x = util.longtensor(self, batch_size, max_len).zero_()
        for n, txt in enumerate(txts):
            for i in range(txt_len[n]): # TODO could this be faster?
                x[n,i] = int(txt[i])
        return self.embed(Varng(x)).view(batch_size, max_len, self.dim), txt_len


class BOWFeatures(macarico.StaticFeatures):
    def __init__(self,
                 n_types,
                 input_field='X',
                 window_size=0,
                 hashing=False):
        dim = (1 + 2 * window_size) * n_types
        macarico.StaticFeatures.__init__(self, dim)
        
        self.n_types = n_types
        self.input_field = input_field
        self.window_size = window_size
        self.hashing = hashing

    def _forward(self, env):
        txt = util.getattr_deep(env, self.input_field)
        bow = util.zeros(self, 1, len(txt), self.dim)
        self.set_bow(bow, 0, txt)
        return Varng(bow)

    def _forward_batch(self, envs):
        batch_size = len(envs)
        txts = [util.getattr_deep(env, self.input_field) for env in envs]
        txt_len = list(map(len, txts))
        max_len = max(txt_len)
        bow = util.zeros(self, batch_size, max_len, self.dim)
        for j, txt in enumerate(txts):
            self.set_bow(bow, j, txt)
        return Varng(bow), txt_len

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


class RNN(macarico.StaticFeatures):
    def __init__(self,
                 features,
                 d_rnn = 50,
                 bidirectional = True,
                 n_layers = 1,
                 cell_type = 'LSTM', # LSTM, GRU, RNN or QRNN (if it's installed)
                 dropout=0.,
                 qrnn_use_cuda=False,  # TODO unfortunately QRNN needs to know this
                 *extra_rnn_args
                 ):
        # model is:
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - dimensionality
        #   n_layers  - how many layers of RNN
        #   bidirectional - is the RNN bidirectional?
        #   cell_type - RNN/GRU/LSTM?
        # we assume that state:Env defines state.N and state.{input_field}
        macarico.StaticFeatures.__init__(self, d_rnn * (2 if bidirectional else 1))

        self.features = features
        self.bidirectional = bidirectional
        self.d_emb = features.dim
        self.d_rnn = d_rnn
        
        assert cell_type in ['LSTM', 'GRU', 'RNN', 'QRNN']
        if cell_type == 'QRNN':
            assert qrnn_available, 'you asked from QRNN but torchqrnn is not installed'
            assert dropout == 0., 'QRNN does not support dropout' # TODO talk to @smerity
            #assert not bidirectional, 'QRNN does not support bidirections, talk to @smerity!'
            self.rnn = QRNN(self.d_emb,
                            self.d_rnn,
                            num_layers=n_layers,
                            use_cuda=qrnn_use_cuda, # TODO do this properly
                            *extra_rnn_args,
                           )
            if bidirectional:
                self.rnn2 = QRNN(self.d_emb,
                                 self.d_rnn,
                                 num_layers=n_layers,
                                 use_cuda=qrnn_use_cuda, # TODO do this properly
                                 *extra_rnn_args,
                                )
                self.rev = list(range(255, -1, -1))
        else:
            self.rnn = getattr(nn, cell_type)(self.d_emb,
                                              self.d_rnn,
                                              num_layers=n_layers,
                                              bidirectional=bidirectional,
                                              dropout=dropout,
                                              batch_first=True,
                                              *extra_rnn_args)

    def _forward(self, env):
        e = self.features(env)
        [res, _] = self.rnn(e)
        if hasattr(self, 'rnn2'):
            [res2, _] = self.rnn2(e[:, self.rev[-e.shape[1]:], :])
            res = torch.cat([res, res2[:, self.rev[-res2.shape[1]:], :]], dim=2)
        return res

    def inv_perm(self, perm):
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse
    
    def _forward_batch(self, envs):
        e, lens = self.features.forward_batch(envs)

        # this is really annoying why can't pytorch do this for us???
        sort_idx = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        e = e[sort_idx]
        sort_lens = [lens[i] for i in sort_idx]
        pack = torch.nn.utils.rnn.pack_padded_sequence(e, sort_lens, batch_first=True)
        [r, _] = self.rnn(e)
        r = r[self.inv_perm(sort_idx)]
        
        if hasattr(self, 'rnn2'):
            # TODO some stuff is padded, don't want to run backward LSTM on it!
            [r2, _] = self.rnn2(e[:, self.rev[-e.shape[1]:], :])
            r = torch.cat([r, r2[:, self.rev[-r2.shape[1]:], :]], dim=2)
        return r, lens


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

    # TODO _forward_batch


class AverageAttention(macarico.Attention):
    arity = None # boil everything down to one item

    def __init__(self, features):
        macarico.Attention.__init__(self, features)
    
    def _forward(self, state):
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
        
    def _forward(self, state):
        x = self.features(state)
        n = self.position(state)
        if n < 0: return [self.oob]
        if n >= x.shape[1]: return [self.oob]
        return [x[0,n].unsqueeze(0)]

    
class FrontBackAttention(macarico.Attention):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __init__(self, features):
        macarico.Attention.__init__(self, features)

    def _forward(self, state):
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

    def _forward(self, state):
        x = self.features(state).squeeze(0)
        h = self.actor[0].hidden()
        N = x.shape[0]
        alpha = self.attention(h.repeat(N,1), x) if self.bilinear else \
                self.attention(torch.cat([h.repeat(N,1), x], 1))
        return [F.softmax(alpha.view(1,N), dim=1).mm(x)]

    # TODO: should we allow attention to precompute? eg bilinear
    # attention of the form xAy could precompute xA (without dynamism)
    # in batch, and then do (xA)h dynamically.
