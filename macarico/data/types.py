from __future__ import division, generators, print_function

import macarico
from macarico.data.vocabulary import NoVocabulary

class Sequences(macarico.Example):
    def __init__(self, tokens, labels, token_vocab, label_vocab):
        if isinstance(token_vocab, int): token_vocab = NoVocabulary(token_vocab)
        if isinstance(label_vocab, int): label_vocab = NoVocabulary(label_vocab)
        self.n_labels = len(label_vocab)
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.tokens = tokens
        self.labels = labels
        self.N = len(tokens)
        
        X = [token_vocab(x) for x in tokens]
        Y = None if labels is None else \
            [label_vocab(x) for x in labels]
        self.N = len(X)
	
        super(Sequences, self).__init__(X, Y)

    def __str__(self):
        words = []
        length = max(len(self.tokens),
                     0 if self.labels is None else len(self.labels),
                     0 if self.Yhat is None else len(self.Yhat))

        for n in range(length):
            w = '</s>' if n >= len(self.tokens) else \
                str(self.tokens[n])
            
            l = '' if self.labels is None else \
                '_</s>' if n >= len(self.labels) else \
                ('_' + str(self.labels[n]))

            p = '' if self.Yhat is None else \
                '_</s>' if n >= len(self.Yhat) else \
                ('_' + str(self.label_vocab(self.Yhat[n])))

            words.append(w + l + p)

        return ' '.join(words)

    def input_str(self):
        return ' '.join(map(str, self.tokens))

    def output_str(self):
        return '?' if self.labels is None else \
               ' '.join(map(str, self.labels))

    def prediction_str(self):
        return '?' if self.Yhat is None else \
               ' '.join([str(self.label_vocab(y)) for y in self.Yhat])
        
class Dependencies(Sequences):
    def __init__(self, tokens, heads, token_vocab, tags=None, tag_vocab=None, rels=None, rel_vocab=None):
        if isinstance(token_vocab, int): token_vocab = NoVocabulary(token_vocab)
        if tags is not None:
            if isinstance(tag_vocab, int): tag_vocab = NoVocabulary(tag_vocab)
            assert len(tags) == len(tokens)
        
        if rels is None:
            self.n_rels = 0
        else:
            if isinstance(rel_vocab, int): rel_vocab = NoVocabulary(rel_vocab)
            self.n_rels = len(rel_vocab)

        self.tokens = tokens
        self.raw_tags = tags
        self.tags = None if tags is None else [tag_vocab(t) for t in tags]
        self.N = len(tokens)
        
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.rel_vocab = rel_vocab

        X = [token_vocab(x) for x in tokens]
        Y = None
        if heads is not None:
            Y = DependencyTree(self.N, rels is not None)
            Y.heads = heads
            if rels is not None:
                Y.rels = [rel_vocab(r) for r in rels]
	
        super(Sequences, self).__init__(X, Y)

    def is_projective(self):
        return self.Y.is_projective()

    def __str__(self):
        return ' ### '.join([self.input_str(), str(self.tags), str(self.raw_tags), self.output_str(), self.prediction_str()])

    def input_str(self):
        return ' '.join([w for w in self.tokens]) if self.tags is None else \
               ' '.join(['%s_%s' % (w,t) for w,t in zip(self.tokens, self.tags)])

    def output_str(self):
        return '?' if self.Y is None else str(self.Y)
    
    def prediction_str(self):
        return '?' if self.Yhat is None else str(self.Yhat)

    
class DependencyTree(object):
    def __init__(self, n, labeled=False):
        self.n = n+1
        self.labeled = labeled
        self.heads = [None] * n
        self.rels = ([None] * n) if labeled else None

    def add(self, head, child, rel=None):
        self.heads[child] = head
        if self.labeled:
            self.rels[child] = rel

    def __getitem__(self, i):
        return self.heads[i], (None if self.rels is None else self.rels[i])
    
    def is_projective(self):
        return not self.is_non_projective()
            
    def is_non_projective(self):
        for dep1, head1 in enumerate(self.heads):
            for dep2, head2 in enumerate(self.heads):
                if head1 < 0 or head2 < 0:
                    continue
                if (dep1 > head2 and dep1 < dep2 and head1 < head2) or \
                   (dep1 < head2 and dep1 > dep2 and head1 < dep2):
                    return True
                if dep1 < head1 and head1 != head2 and \
                   ((head1 > head2 and head1 < dep2 and dep1 < head2) or \
                    (head1 < head2 and head1 > dep2 and dep1 < dep1)):
                    return True
        return False
            
    def __repr__(self):
        s = 'heads = %s' % str(self.heads)
        if self.labeled:
            s += '\nrels  = %s' % str(self.rels)
        return s

    def __str__(self):
        S = []
        for i in range(self.n-1):
            x = '%d->%s' % (i, self.heads[i])
            if self.labeled:
                x = '%s[%s]' % (x, self.rels[i])
            S.append(x)
        return ' '.join(S)
            
