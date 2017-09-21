from __future__ import division

import sys
from collections import Counter
import gzip
import numpy as np
from macarico.tasks import dependency_parser as dp
from macarico.tasks import sequence_labeler as sl
from macarico.tasks import seq2seq as s2s
import macarico 
import codecs

def read_underscore_tagged_text(filename, max_examples=None):
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
            if max_examples is not None and len(data) >= max_examples:
                break

    for x in data:
        x.n_labels = len(label_id)

    return data, label_id


def read_embeddings(filename, vocab):
    emb = None
    my_open = gzip.open if filename.endswith('.gz') else open
    with my_open(filename, 'r') as h:
        for l in h.readlines():
            a = l.strip().split()
            w = a[0]
            if emb is None:
                emb = np.random.randn(len(vocab), len(a)-1)
            if w in vocab:
                a = np.array(map(float, a[1:]))
                emb[vocab[w],:] = a / a.std()
    return emb

def read_conll_dependecy_text(filename, labeled, max_examples=None, max_length=None):
    with open(filename) as h:
        data = []
        rel_id = {}
        new = lambda : dp.Example([], pos=[], heads=[],
                                  rels=[] if labeled else None,
                                  n_rels=None)
        example = new()
        for l in h:
            a = l.strip().split()
            if len(a) == 0:
                if len(example.tokens) > 0 and (max_length is None or len(example.tokens) <= max_length):
                    data.append(example)
                if max_examples is not None and len(data) > max_examples:
                    break
                example = new()
                continue
            [w,t,h,r] = a
            example.tokens.append(w)
            example.pos.append(t)
            h = int(h)
            example.heads.append(h if h >= 0 else None)
            if labeled:
                if r not in rel_id:
                    rel_id[r] = len(rel_id)
                example.rels.append(rel_id[r])
        if len(example.tokens) > 0 and (max_length is None or len(example.tokens) <= max_length):
            data.append(example)

        for x in data:
            # rewrite None as head as n
            x.heads = [h or len(x.tokens) for h in x.heads]
            # set n_rels only after we know it.
            if labeled:
                x.n_rels = len(rel_id)

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
    vocab = {EOS: 0, BOS: 1, OOV: 2}
    for token, count in counts.iteritems():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def apply_vocab(vocab, data, dim, lowercase):
    def f(x):
        if isinstance(x,str) and x not in SPECIAL and lowercase: x = x.lower()
        return vocab.get(x, vocab[OOV])
    for x in data:
        setattr(x, dim, map(f, getattr(x, dim)))


def read_wsj_pos(filename, n_tr=20000, n_de=2000, n_te=3859, min_freq=5, lowercase=True):
    data, label_id = read_underscore_tagged_text(filename, n_tr+n_de+n_te)
    tr = data[:n_tr]
    token_vocab = build_vocab(tr, 'tokens', min_freq, lowercase=lowercase)
    apply_vocab(token_vocab, data, 'tokens', lowercase=lowercase)
    return (tr,
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            token_vocab,
            label_id)


def read_wsj_deppar(filename='data/deppar.txt', n_tr=39829, n_de=1700,
                    n_te=2416, min_freq=5, lowercase=True,
                    labeled=False, max_length=None):

    data, rel_id = read_conll_dependecy_text(filename, labeled,
                                             n_tr+n_de+n_te, max_length)
    tr = data[:n_tr]

    # build vocab on train.
    word_vocab = build_vocab(tr, 'tokens', min_freq, lowercase=lowercase)
    pos_vocab = build_vocab(tr, 'pos', min_freq=0, lowercase=False)

    # apply vocab to all of the data
    apply_vocab(word_vocab, data, 'tokens', lowercase=lowercase)
    apply_vocab(pos_vocab , data, 'pos', lowercase=False)

    return (data[:n_tr],
            data[n_tr:n_tr+n_de],
            data[n_tr+n_de:],
            word_vocab,
            pos_vocab,
            rel_id)

def read_bilingual_pairs(src_filename, tgt_filename, max_src_len, max_tgt_len, max_ratio):
    with codecs.open(src_filename, encoding='utf-8') as src_h:
        with codecs.open(tgt_filename, encoding='utf-8') as tgt_h:
            data = []
            for src_l in src_h:
                e = tgt_h.readline().strip().split()
                f = src_l.strip().split()
                if max_src_len is not None and len(f) > max_src_len: continue
                if max_tgt_len is not None and len(e) > max_tgt_len: continue
                if len(e) == 0 or len(f) == 0: continue
                if max_ratio is not None:
                    ratio = len(e) / len(f)
                    if ratio > max_ratio or 1/ratio > max_ratio: continue
                data.append(s2s.Example(f, e, n_labels=None))
    return data

def read_parallel_data(src_filename, tgt_filename, n_de=2000,
                       min_src_freq=5, min_tgt_freq=None,
                       lowercastgt_f=True, lowercastgt_e=None,
                       max_src_len=None, max_tgt_len=None, max_ratio=None,
                       remove_tgt_oov=True, shuffle=False):
    min_tgt_freq = min_tgt_freq if min_tgt_freq is not None else min_src_freq
    lowercastgt_e = lowercastgt_e if lowercastgt_e is not None else lowercastgt_f
    data = read_bilingual_pairs(src_filename, tgt_filename, max_src_len, max_tgt_len, max_ratio)
    if shuffle:
        np.random.shuffle(data)
    src_vocab = build_vocab(data, 'tokens', min_src_freq, lowercase=lowercastgt_f)
    tgt_vocab = build_vocab(data, 'labels', min_tgt_freq, lowercase=lowercastgt_e)
    apply_vocab(src_vocab, data, 'tokens', lowercase=lowercastgt_f)
    apply_vocab(tgt_vocab, data, 'labels', lowercase=lowercastgt_e)
    n_labels = len(tgt_vocab)
    tgt_oov = tgt_vocab[OOV]
    for ex in data:
        ex.n_labels = n_labels
        if remove_tgt_oov:
            ex.original_labels = ex.labels + [0]
            ex.labels = [e for e in ex.labels if e != tgt_oov]
        ex.labels += [0]
    n_tr = len(data) - n_de
    return (data[:n_tr], data[n_tr:], src_vocab, tgt_vocab)

def ngrams(words):
    c = Counter()
    for l in range(4):
        for ng in zip(*[words]*(l+1)):
            c[ng] += 1
    return c

class Bleu(macarico.Loss):
    def __init__(self):
        super(Bleu, self).__init__('bleu', corpus_level=True)
        self.sys = np.zeros(4)
        self.cor = np.zeros(4)
        self.len_sys = 0
        self.len_ref = 0

    def reset(self):
        self.sys = np.zeros(4)
        self.cor = np.zeros(4)
        self.len_sys = 0
        self.len_ref = 0
        
    def evaluate(self, truth, state):
        prediction = state.output
        labels = truth.original_labels if hasattr(truth, 'original_labels') else \
                 truth.labels
        assert labels[-1] == 0  # </s>
        self.len_ref += len(labels) - 1
        self.len_sys += len(prediction)

        ref = ngrams(labels[:-1])
        sys = ngrams(prediction)
        for ng, count in sys.iteritems():
            l = len(ng)-1
            self.sys[l] += count
            self.cor[l] += min(count, ref[ng])

        precision = self.cor / (self.sys + 1e-6)
        brev = min(1., np.exp(1 - self.len_ref / self.len_sys)) if self.len_sys > 0 else 0
        return 1 - brev * precision.prod()
