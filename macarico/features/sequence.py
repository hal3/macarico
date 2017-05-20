from __future__ import division

import torch
from torch import nn
from torch.autograd import Variable

import macarico

class RNNFeatures(macarico.Features, nn.Module):

    def __init__(self,
                 n_types,
                 input_field = 'tokens',
                 output_field = 'tokens_feats',
                 d_emb = 50,
                 d_rnn = 50,
                 bidirectional = True,
                 n_layers = 1,
                 rnn_type = nn.LSTM,
                 initial_embeddings = None,
                 learn_embeddings = True):
        # model is:
        #   embed words using standard embeddings, e[n]
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - dimensionality
        #   n_layers  - how many layers of RNN
        #   bidirectional - is the RNN bidirectional?
        #   rnn_type - RNN/GRU/LSTM?
        # we assume that state:Env defines state.N and state.{input_field}

        nn.Module.__init__(self)

        self.input_field = input_field

        if d_emb is None and initial_embeddings is not None:
            d_emb = initial_embeddings.shape[1]
        
        self.d_emb = d_emb
        self.d_rnn = d_rnn
        self.embed_w = nn.Embedding(n_types, self.d_emb)
        self.rnn = rnn_type(self.d_emb,
                            self.d_rnn,
                            num_layers = n_layers,
                            bidirectional = bidirectional)

        if not learn_embeddings:
            self.embed_w.weight.requires_grad = False # don't train embeddings

        if initial_embeddings is not None:
            e0_v, e0_d = initial_embeddings.shape
            assert e0_v == n_types, \
                'got initial_embeddings with first dim=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, \
                'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            self.embed_w.weight.data.copy_(torch.from_numpy(initial_embeddings))
            

        macarico.Features.__init__(self, output_field, d_rnn * (2 if bidirectional else 1))

    def _forward(self, state):
        # run a BiLSTM over input on the first step.
        my_input = getattr(state, self.input_field)
        e = self.embed_w(Variable(torch.LongTensor(my_input)))
        [res, _] = self.rnn(e.view(state.N,1,-1))
        return res
    
class BOWFeatures(macarico.Features, nn.Module):

    def __init__(self,
                 n_types,
                 input_field  = 'tokens',
                 output_field = 'tokens_feats',
                 window_size  = 0,
                 max_length   = 255):
        nn.Module.__init__(self)  # TODO: is this necessary?
        
        self.n_types = n_types
        self.input_field = input_field
        self.window_size = window_size
        self.max_length = max_length

        dim = (1 + 2 * window_size) * n_types
        self.onehots = {}

        macarico.Features.__init__(self, output_field, dim)

    #@profile
    def _forward(self, state):
        # this version takes 44 seconds
        my_input = getattr(state, self.input_field)
        output = torch.zeros(len(my_input), 1, self.dim)
        for n, w in enumerate(my_input):
            if w not in self.onehots:
                data = torch.zeros(1, self.dim)
                data[0,w] = 1.
                self.onehots[w] = data
            output[n,0,:] = self.onehots[w]

        return Variable(output, requires_grad=False)

class AverageAttention(macarico.Attention):
    arity = None # boil everything down to one item

    def __init__(self, field='tokens_feats'):
        super(AverageAttention, self).__init__(field)
    
    def __call__(self, state):
        N = state.N
        return Variable(torch.ones(1,N) / N, requires_grad=False)

#inp = torch.LongTensor(16, 28) % n    
#inp_ = torch.unsqueeze(inp, 2)
#one_hot = torch.FloatTensor(16, 28, n).zero_()
#one_hot.scatter_(2, inp_, 1)

class AttendAt(macarico.Attention):
    """Attend to the current token's *input* embedding.

    TODO: We should be able to attend to the *output* embeddings too, i.e.,
    embedding of the previous actions and hidden states.

    TODO: Will need to cover boundary token embeddings in some reasonable way.

    """
    arity = 1
    def __init__(self,
                 get_position=lambda state: state.n,
                 field='tokens_feats'):
        self.get_position = get_position
        super(AttendAt, self).__init__(field)

    def __call__(self, state):
        return [self.get_position(state)]

class FrontBackAttention(macarico.Attention):
    """
    Attend to front and end of input string; if run with a BiLStM
    (eg), this should be sufficient to capture whatever you want.
    """
    arity = 2
    def __init__(self, field='tokens_feats'):
        super(FrontBackAttention, self).__init__(field)

    def __call__(self, state):
        return [0, state.N-1]

class SoftmaxAttention(macarico.Attention, nn.Module):
    arity = None  # attention everywhere!
    
    def __init__(self, input_features, d_state, hidden_state='h'):
        nn.Module.__init__(self)

        self.input_features = input_features
        self.d_state = d_state
        self.hidden_state = hidden_state
        self.d_input = input_features.dim + d_state
        self.mapping = nn.Linear(self.d_input, 1)
        self.softmax = nn.Softmax()

        macarico.Attention.__init__(self, input_features.field)

    def __call__(self, state):
        N = state.N
        fixed_inputs = self.input_features(state)
        hidden_state = getattr(state, self.hidden_state)[state.t-1] if state.t > 0 else \
                       getattr(state, self.hidden_state + '0')
        #print fixed_inputs
        output = torch.cat([fixed_inputs.squeeze(1), hidden_state.repeat(N,1)], 1)
        return self.softmax(self.mapping(output)).view(1,-1)

