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
            data.append([tokens, labels])
    return data, label_id


def read_conll_dependecy_text(filename):
    rel_id = {}
    data = []
    warned = False
    with open(filename, 'r') as h:
        words,tags,heads,rels = [],[],[],[]
        for l in h.readlines():
            a = l.strip().split()
            if len(a) == 0:
                if len(words) > 0:
                    data.append([words,tags,heads,rels])
                words,tags,heads,rels = [],[],[],[]
            elif len(a) == 4 and (a[2].isdigit() or
                                  (len(a[2]) > 1 and
                                   a[2][0] == '-' and
                                   a[2][1:].isdigit())):
                words.append(a[0])
                tags.append(a[1])
                hd = int(a[2])
                heads.append(hd if hd >= 0 else None)
                if a[3] not in rel_id:
                    rel_id[a[3]] = len(rel_id)
                rels.append(rel_id[a[3]])
            elif not warned:
                print >>sys.stderr, 'warning: malformed tab separated line "%s" (suppressing future warning)' % l.strip()
                warned = True
        if len(words) > 0:
            data.append([words,tags,heads,rels])
    return data, rel_id


def build_vocab(sentences, min_word_freq=5, lowercase=True):
    counts = Counter()
    for tokens in sentences:
        for token in tokens:
            if lowercase:
                token = token.lower()
            counts[token] += 1
    if lowercase:
        vocab = {'*oov*': 0}
    else:
        vocab = {'*OOV*': 0}
    for token,count in counts.iteritems():
        if count >= min_word_freq:
            vocab[token] = len(vocab)
    return vocab


def apply_vocab(vocab, data, dim=0):
    lowercase = '*oov*' in vocab
    def apply_vocab2(w):
        if lowercase: w = w.lower()
        return vocab.get(w, 0)
    for i in xrange(len(data)):
        data[i][dim] = map(apply_vocab2, data[i][dim])


# XXX: currently ignoring lowercase option
def read_wsj_pos(filename, n_tr=20000, n_de=2000,
                 min_word_freq=5, lowercase=True):
    data, label_id = read_underscore_tagged_text(filename)
    tokens, _ = zip(*data[:n_tr])
    token_vocab = build_vocab(tokens, min_word_freq)
    apply_vocab(token_vocab, data, 0)
    return (data[:n_tr],
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            token_vocab,
            label_id)


# XXX: currently ignoring lowercase option
def read_wsj_deppar(filename='data/deppar.txt', n_tr=39829, n_de=1700,
                    min_word_freq=5, lowercase=True):
    data, rel_id = read_conll_dependecy_text(filename)
    [tokens, pos, _, _] = zip(*data[:n_tr])
    word_vocab = build_vocab(tokens, min_word_freq)
    pos_vocab = build_vocab(pos, 0)
    apply_vocab(word_vocab, data, 0)
    apply_vocab(pos_vocab , data, 1)
    return (data[:n_tr],
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            word_vocab,
            pos_vocab,
            rel_id)
