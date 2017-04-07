import macarico as lts

class SequenceLabeler(lts.SearchTask):
    def __init__(self, n_words, n_labels, ref, **kwargs):
        # we need to know the size of the hidden state
        n_hid    = kwargs.get('n_hid',    50)
        n_emb    = kwargs.get('n_emb',    n_hid)
        n_layers = kwargs.get('n_layers', 1)
        
        # initialize the parent class; this needs to know the
        # branching factor of the task (in this case, the branching
        # factor is exactly the number of labels). it also needs to
        # know the dimensionality of the thing that will be used to
        # make that prediction.
        super(SequenceLabeler, self).__init__(n_hid, n_labels, ref)

        # set up simple sequence labeling model, which runs an LSTM
        # _backwards_ over the input, and then predicts left-to-right
        self.encoder = nn.Embedding( n_words, n_emb )
        self.rnn     = nn.LSTM(n_emb, n_hid, n_layers, dropout=kwargs.get('dropout', 0.5))

    def _run(self, words, labels):
        # if labels is None, then it's a test example;
        # otherwise we must have as many labels as words!
        assert labels is None or len(words) == len(labels),
          'error: on a training example, |labels|=%d must equal |words|=%d' %
          (len(words), len(labels))

        N = len(words)

        # set up the reference policy
        self.ref_init(labels)
        
        # first, run the LSTM backwords over (embeddings of) words
        embeddings = self.encoder(words)
        _,hiddens  = self.rnn(embeddings, self._init_hidden())
        
        # second, make predictions left-to-right
        output = []
        for n in range(N):
            # extract the neural state on which we will make a
            # prediction; since the sentence was encoded backwards, we
            # read this from the end
            state = hiddens[N-n-1]
            
            # choose an action by calling self.act; this is defined
            # for you by lts.SearchTask
            a = self.act(state)

            # append output, and tell ref what we predicted
            output.append(a)

        return output

    def _init_hidden(self):
        return Variable(torch.zeros(1, 1, self.n_hid))
