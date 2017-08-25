from __future__ import division

#import torch
#from torch import nn
#from torch.autograd import Variable
import numpy as np
import dynet as dy
import macarico

def getattr_deep(obj, field):
    for f in field.split('.'):
        obj = getattr(obj, f)
    return obj

class RNNFeatures(macarico.Features):

    def __init__(self,
                 dy_model,
                 n_types,
                 input_field = 'tokens',
                 output_field = 'tokens_feats',
                 d_emb = 50,
                 d_rnn = 50,
                 bidirectional = True,
                 n_layers = 1,
                 rnn_type = 'LSTM', # LSTM, GRU or None
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

        #nn.Module.__init__(self)
        self.dy_model = dy_model

        self.input_field = input_field
        self.bidirectional = bidirectional

        if d_emb is None and initial_embeddings is not None:
            d_emb = initial_embeddings.shape[1]

        self.d_emb = d_emb
        self.d_rnn = d_rnn
        #self.embed_w = nn.Embedding(n_types, self.d_emb)

        self.learn_embeddings = learn_embeddings
        
        if initial_embeddings is not None:
            e0_v, e0_d = initial_embeddings.shape
            assert e0_v == n_types, \
                'got initial_embeddings with first dim=%d != %d=n_types' % (e0_v, n_types)
            assert e0_d == d_emb, \
                'got initial_embeddings with second dim=%d != %d=d_emb' % (e0_d, d_emb)
            #self.embed_w.weight.data.copy_(torch.from_numpy(initial_embeddings))

        if learn_embeddings:
            if initial_embeddings is not None:
                initial_embeddings = dy.NumpyInitializer(initial_embeddings)
            self.embed_w = dy_model.add_lookup_parameters((n_types, self.d_emb), initial_embeddings)
        else:
            assert initial_embeddings is not None
            self.embed_w = initial_embeddings

        rnn_builder = getattr(dy, rnn_type + "Builder")
        self.f_rnn = rnn_builder(1, self.d_emb, self.d_rnn, dy_model)
        self.b_rnn = rnn_builder(1, self.d_emb, self.d_rnn, dy_model) if bidirectional else None

        macarico.Features.__init__(self, output_field, self.d_rnn * (2 if bidirectional else 1))

#    @profile
    def _forward(self, state):
        # run a BiLSTM over input on the first step.
        my_input = getattr_deep(state, self.input_field)
        embed = [self.embed_w[w] for w in my_input]
        if not self.learn_embeddings:
            embed = map(dy.inputTensor, embed)
        f_emb = self.f_rnn.initial_state().transduce(embed)
        if self.bidirectional:
            b_emb = reversed(self.f_rnn.initial_state().transduce(reversed(embed)))
            f_emb = [dy.concatenate([f, b]) for f, b in zip(f_emb, b_emb)]
        
        #if self.rnn is not None:
        #    [res, _] = self.rnn(e)
        #else:
        #    res = e
        return f_emb

class BOWFeatures(macarico.Features):

    def __init__(self,
                 dy_model,
                 n_types,
                 input_field  = 'tokens',
                 output_field = 'tokens_feats',
                 window_size = 0):
        self.dy_model = dy_model
        
        self.n_types = n_types
        self.input_field = input_field
        self.window_size = window_size

        dim = (1 + 2 * window_size) * n_types
        macarico.Features.__init__(self, output_field, dim)

    #@profile
    def _forward(self, state):
        # this version takes 44 seconds
        my_input = getattr_deep(state, self.input_field)
        output = np.zeros((len(my_input), self.dim))
        for n, w in enumerate(my_input):
            for i in range(-self.window_size, self.window_size+1):
                m = n + i
                if m < 0: continue
                if m >= len(my_input): continue
                v = (i + self.window_size) * self.n_types + w
                output[m, v] = 1
#                
#            if w not in self.onehots:
#                data = torch.zeros(1, self.dim)
#                data[0,w] = 1.
#                self.onehots[w] = data
#            output[n,0,:] = self.onehots[w]

        return dy.inputTensor(output) # Variable(output, requires_grad=False)

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

"""
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
"""
