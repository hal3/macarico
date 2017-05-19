from __future__ import division

import torch
from torch import nn
from torch.autograd import Variable

import macarico

class RNNFeatures(macarico.Features, nn.Module):

    def __init__(self,
                 n_types,
                 input_field = 'tokens',
                 output_field = 'tokens_rnn',
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

        nn.Module.__init__(self)

        self.input_field = input_field
        self.output_field = output_field

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
            

        macarico.Features.__init__(self, d_rnn * (2 if bidirectional else 1))

    def forward(self, state):
        if not hasattr(state, self.output_field) or \
           getattr(state, self.output_field) is None:
            # run a BiLSTM over input on the first step.
            my_input = getattr(state, self.input_field)
            e = self.embed_w(Variable(torch.LongTensor(my_input)))
            [res, _] = self.rnn(e.view(state.N,1,-1))
            setattr(state, self.output_field, res)

        return getattr(state, self.output_field)
    
class BOWFeatures(macarico.Features, nn.Module):

    def __init__(self,
                 n_types,
                 input_field  = 'tokens',
                 output_field = 'tokens_bow',
                 window_size  = 0,
                 max_length   = 255):
        nn.Module.__init__(self)  # TODO: is this necessary?
        
        self.n_types = n_types
        self.input_field = input_field
        self.output_field = output_field
        self.window_size = window_size
        self.max_length = max_length

        dim = (1 + 2 * window_size) * n_types
        self.onehots = {}

        macarico.Features.__init__(self, dim)

    #@profile
    def forward(self, state):
        if not hasattr(state, self.output_field) or \
               getattr(state, self.output_field) is None:
            # this version takes 44 seconds
            my_input = getattr(state, self.input_field)
            output = torch.zeros(len(my_input), 1, self.dim)
            for n, w in enumerate(my_input):
                if w not in self.onehots:
                    data = torch.zeros(1, self.dim)
                    data[0,w] = 1.
                    self.onehots[w] = data
                output[n,0,:] = self.onehots[w]
            setattr(state, self.output_field, Variable(output, requires_grad=False))

        return getattr(state, self.output_field)
    
#inp = torch.LongTensor(16, 28) % n    
#inp_ = torch.unsqueeze(inp, 2)
#one_hot = torch.FloatTensor(16, 28, n).zero_()
#one_hot.scatter_(2, inp_, 1)
