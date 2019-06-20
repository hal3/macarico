from __future__ import division, generators, print_function

import macarico.util as util
import macarico.tasks.mdp as mdp
from macarico.data.types import Sequences
from macarico.data.vocabulary import EOS
import macarico.tasks.seq2json as s2j

import numpy as np

########################################################
# synthetic data construction

def make_sequence_reversal_data(num_ex, ex_len, n_types, include_eos=False):
    data = []
    low = 1 if include_eos else 0
    for _ in range(num_ex):
        x = [int(np.random.choice(range(low, n_types))) for _ in range(ex_len)]
        y = list(reversed(x))
        if include_eos:
            x += [EOS]
            y += [EOS]
        data.append(Sequences(x, y, n_types, n_types))
    return data

def make_sequence_mod_data(num_ex, ex_len, n_types, n_labels, include_eos=False):
    if not isinstance(ex_len, list): ex_len = [ex_len]
    data = []
    low = 1 if include_eos else 0
    for _ in range(num_ex):
        x = [int(low+i) for i in np.random.randint(n_types-low, size=np.random.choice(ex_len))]
        y = [low+((i+1) % (n_labels-low)) for i in x]
        if include_eos:
            x += [EOS]
            y += [EOS]
        data.append(Sequences(x, y, n_types, n_labels))
    return data

def make_json_mod_data(num_ex, ex_len, n_types, n_labels):
    if not isinstance(ex_len, list): ex_len = [ex_len]
    assert n_labels >= 3
    max_n = max(n_types, n_labels)
    rand = lambda: np.random.randint(max_n-1)
    def make_json_mod_data_rec(len_remain):
        if len_remain <= 1:
            return rand(), 0
        typ = np.random.randint(3)
        if typ == 0:
            return rand(), len_remain-1
        if typ == 1:
            ll = np.random.randint(3) + 1
            this = []
            for _ in range(ll):
                x, c = make_json_mod_data_rec(len_remain)
                len_remain -= c
                this.append(x)
                if len_remain <= 0: break
            return this, len_remain
        if typ == 2:
            ll = np.random.randint(2) + 1
            this = {}
            for _ in range(ll):
                k = rand()
                v, c = make_json_mod_data_rec(len_remain)
                len_remain -= c
                this[k] = v
                if len_remain <= 0: break
            return this, len_remain
        raise Exception()
    def flatten_json(t, k, v):
        return t + [1 if k else 2, (v % n_types)+1]
    data = []
    for _ in range(num_ex):
        json, _ = make_json_mod_data_rec(np.random.choice(ex_len))
        tok = s2j.fold_json(flatten_json,
                            [],
                            json)
        tok += [EOS]
        ex = s2j.Seq2JSONExample(tok, s2j.map_json(lambda k: k % n_labels, json))
        #print('ex=', ex.tokens, ex.truth)
        data.append(ex)
    return data

def make_ross_mdp(T=100, reset_prob=0):
    initial = [(0, 1/3), (1, 1/3)]
    #               s    a    s' p()
    half_rp = reset_prob/2
    default = 1-reset_prob
    transitions = { 0: { 0: [(1, default), (0, half_rp), (2, half_rp)],
                         1: [(2, default), (0, half_rp), (1, half_rp)] },
                    1: { 0: [(2, default), (0, half_rp), (1, half_rp)],
                         1: [(1, default), (0, half_rp), (2, half_rp)] },
                    2: { 0: [(1, default), (1, half_rp), (2, half_rp)],
                         1: [(2, default), (0, half_rp), (2, half_rp)] } }

    def pi_ref(s):
        if isinstance(s, mdp.MDP):
            s = s.s
        # expert: s0->a0 s1->a1 s2->a0
        if s == 0: return 0
        if s == 1: return 1
        if s == 2: return 0
        assert False
        
    def costs(s, a, s1):
        # this is just Cmax=1 whenever we disagree with expert, and c=0 otherwise
        return 0 if a == pi_ref(s) else 1
    
    return mdp.MDP(mdp.MDPExample(initial, transitions, costs, T)), \
           mdp.DeterministicReference(pi_ref)
