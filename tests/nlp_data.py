import sys
from collections import Counter

def read_underscore_tagged_text(filename):
    label_id = {}
    data = []
    warned = False
    with open(filename, 'r') as h:
        for l in h.readlines():
            token_labels = l.strip().split()
            if len(token_labels) == 0:
                continue
            tokens = []
            labels = []
            for token_label in token_labels:
                i = token_label.rfind('_', 0, -1)  # don't allow empty labels
                if i <= 0:
                    if not warned:
                        print >>sys.stderr, 'warning: malformed underscore-separated word "%s" (suppressing future warning)' % token_label
                        warned = True
                    token = token_label
                    label = 'UNK'
                else:
                    token = token_label[:i]
                    label = token_label[i+1:]

                if label not in label_id:
                    label_id[label] = len(label_id)
                tokens.append(token)
                labels.append(label_id[label])
            data.append( (tokens, labels) )
    return data, label_id

def build_vocab(data, min_word_freq=5, lowercase=True):
    counts = Counter()
    for tokens,_ in data:
        for token in tokens:
            if lowercase:
                token = token.lower()
            counts[token] += 1
    if lowercase:
        vocab = { '*oov*': 0 }
    else:
        vocab = { '*OOV*': 0 }
    for token,count in counts.iteritems():
        if count >= min_word_freq:
            vocab[token] = len(vocab)
    return vocab

def apply_vocab(vocab, data):
    lowercase = '*oov*' in vocab
    def apply_vocab2(w):
        if lowercase: w = w.lower()
        return vocab.get(w, 0)
    return [(map(apply_vocab2, tokens), labels) for tokens,labels in data]

def read_wsj(filename, n_tr=20000, n_de=2000, min_word_freq=5, lowercase=True):
    data, label_id = read_underscore_tagged_text(filename)
    vocab = build_vocab(data[:n_tr], min_word_freq, lowercase)
    return apply_vocab(vocab, data[:n_tr]), \
           apply_vocab(vocab, data[n_tr:n_tr+n_de]), \
           apply_vocab(vocab, data[n_tr+n_de:]), \
           vocab, \
           label_id
