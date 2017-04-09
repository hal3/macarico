import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .. import macarico

class SequenceLabeler(macarico.SearchTask):
    def __init__(self, n_words, n_labels, ref_policy, **kwargs):
        # model is:
        #   embed words using standard embeddings, e[n]
        #   run biLSTM backwards over e[n], get r[n] = biLSTM state
        #   h[-1] = zero
        #   for n in range(N):
        #     ae   = embed_action(y[n-1]) or zero if n=0
        #     h[n] = combine(r[n], ae, h[n-1])
        #     y[n] = act(h[n])
        # we need to know dimensionality for:
        #   d_emb     - word embedding e[]
        #   d_rnn     - RNN state r[]
        #   d_actemb  - action embeddings p[]
        #   d_hid     - hidden state
        self.d_emb    = kwargs.get('d_emb',    50)
        self.d_rnn    = kwargs.get('d_rnn',    self.d_emb)
        self.d_actemb = kwargs.get('d_actemb', 5)
        self.d_hid    = kwargs.get('d_hid',    self.d_emb)
        
        # initialize the parent class; this needs to know the
        # branching factor of the task (in this case, the branching
        # factor is exactly the number of labels), the dimensionality
        # of the thing that will be used to make that prediction, and
        # the reference policy. we tell the search task to
        # automatically handle the reference policy for us. this
        # _only_ works when there is a one-to-one mapping between our
        # output and the sequence of actions we take; otherwise we
        # would have to handle the reference policy on our own.
        super(SequenceLabeler, self).__init__(self.d_hid,
                                              n_labels,
                                              ref_policy,
                                              autoref=True)

        # set up simple sequence labeling model, which runs a biRNN
        # over the input, and then predicts left-to-right
        self.embed_w = nn.Embedding(n_words, self.d_emb)
        self.rnn     = nn.RNN(self.d_emb, self.d_rnn, 1,   # 1 is n_layers
                              bidirectional=True,
                              dropout=kwargs.get('dropout', 0.5))
        self.embed_a = nn.Embedding(n_labels, self.d_actemb)
        self.combine = nn.Linear(self.d_rnn*2 + self.d_actemb + self.d_hid,
                                 self.d_hid)

    def _run(self, words):
        # a few silly helper functions to make things cleaner
        zeros  = lambda d: Variable(torch.zeros(1,d))
        onehot = lambda i: Variable(torch.LongTensor([i]))
        
        N = len(words)
        
        # run the LSTM over (embeddings of) words
        e   = self.embed_w(words)
        r,_ = self.rnn(e.view(N,1,-1))
        
        # make predictions left-to-right
        output = []
        h      = zeros(self.d_hid)
        for n in range(N):
            # embed the previous action (if it exists)
            ae = zeros(self.d_actemb)                   if n == 0 \
                 else self.embed_a(onehot(output[n-1]))
            
            # combine hidden state appropriately
            h = F.tanh( self.combine( torch.cat([r[n], ae, h], 1) ) )

            # choose an action by calling self.act; this is defined
            # for you by macarico.SearchTask
            a = self.act(h)

            # append output
            output.append(a)

        return output
        
    

