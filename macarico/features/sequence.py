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
                 rnn_type = nn.LSTM):
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
        self.d_emb = d_emb
        self.d_rnn = d_rnn
        self.embed_w = nn.Embedding(n_types, self.d_emb)
        self.rnn = rnn_type(self.d_emb,
                            self.d_rnn,
                            num_layers = n_layers,
                            bidirectional = bidirectional)

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
