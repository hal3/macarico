import sys
from collections import Counter
from macarico.tasks import dependency_parser as dp
from macarico.tasks import sequence_labeler as sl


def read_underscore_tagged_text(filename):
    label_id = {}
    data = []
    warned = False
    new = lambda : sl.Example([], [], n_labels=None)
    with open(filename, 'r') as h:
        for l in h:
            token_labels = l.strip().split()
            if len(token_labels) == 0:
                continue

            example = new()
            for token_label in token_labels:
                i = token_label.rfind('_', 0, -1)  # don't allow empty labels
                if i <= 0:
                    if not warned:
                        print >>sys.stderr, 'warning: malformed underscore-separated word "%s" (suppressing future warning)' % token_label
                        warned = True
                    token = token_label
                    label = OOV
                else:
                    token = token_label[:i]
                    label = token_label[i+1:]

                if label not in label_id:
                    label_id[label] = len(label_id)
                example.tokens.append(token)
                example.labels.append(label_id[label])

            data.append(example)

    for x in data:
        x.n_labels = len(label_id)

    return data, label_id


def read_conll_dependecy_text(filename):
    with open(filename) as h:
        data = []
        rel_id = {}
        new = lambda : dp.Example([], pos=[], heads=[], rels=[], n_rels=None)
        example = new()
        for l in h:
            a = l.strip().split()
            if len(a) == 0:
                if len(example.tokens) > 0:
                    data.append(example)
                example = new()
                continue
            [w,t,h,r] = a
            example.tokens.append(w)
            example.pos.append(t)
            h = int(h)
            example.heads.append(h if h >= 0 else None)
            if r not in rel_id:
                rel_id[r] = len(rel_id)
            example.rels.append(rel_id[r])
        if len(example.tokens) > 0:   # in case there is no newline at the end of the file.
            data.append(example)

        for x in data:
            x.n_rels = len(rel_id)   # set n_rels only after we know it.

        return data, rel_id


# TODO: Other good OOV strategies
#  - map work to it's longest frequent suffix,
#  - Berkeley's OOV rules
# XXX: only token VOCAB should have these symbols. Don't add to tags!
OOV = '<OOV>'
BOS = '<s>'
EOS = '</s>'
SPECIAL = {OOV, BOS, EOS}
def build_vocab(sentences, field, min_freq=0, lowercase=False):
    counts = Counter()
    for e in sentences:
        for x in getattr(e, field):
            if x not in SPECIAL and lowercase: x = x.lower()
            counts[x] += 1
    vocab = {OOV: 0, BOS: 1, EOS: 2}
    for token, count in counts.iteritems():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def apply_vocab(vocab, data, dim, lowercase):
    def f(x):
        if x not in SPECIAL and lowercase: x = x.lower()
        return vocab.get(x, vocab[OOV])
    for x in data:
        setattr(x, dim, map(f, getattr(x, dim)))


def read_wsj_pos(filename, n_tr=20000, n_de=2000, min_freq=5, lowercase=True):
    data, label_id = read_underscore_tagged_text(filename)
    tr = data[:n_tr]
    token_vocab = build_vocab(tr, 'tokens', min_freq, lowercase=lowercase)
    apply_vocab(token_vocab, data, 'tokens', lowercase=lowercase)
    return (tr,
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            token_vocab,
            label_id)


def read_wsj_deppar(filename='data/deppar.txt', n_tr=39829, n_de=1700,
                    min_freq=5, lowercase=True):

    data, rel_id = read_conll_dependecy_text(filename)
    tr = data[:n_tr]

    # build vocab on train.
    word_vocab = build_vocab(tr, 'tokens', min_freq, lowercase=lowercase)
    pos_vocab = build_vocab(tr, 'pos', min_freq=0, lowercase=False)

    # apply vocab to all of the data!
    apply_vocab(word_vocab, data, 'tokens', lowercase=lowercase)
    apply_vocab(pos_vocab , data, 'pos', lowercase=False)

    return (tr,
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            word_vocab,
            pos_vocab,
            rel_id)
