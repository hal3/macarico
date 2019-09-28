import sys
import random
import macarico
import zss
from macarico.data.vocabulary import EOS

def fold_json(f, x0, json):
    if isinstance(json, int):
        return f(x0, True, json)
    if isinstance(json, list):
        for x in json:
            x0 = fold_json(f, x0, x)
        return x0
    if isinstance(json, dict):
        for k, v in json.items():
            x0 = f(x0, False, k)
            x0 = fold_json(f, x0, v)
        return x0
    assert False

def map_json(f, json):
    if isinstance(json, int):
        return f(json)
    if isinstance(json, list):
        return [map_json(f, j) for j in json]
    if isinstance(json, dict):
        return { f(k): map_json(f, j) for k, j in json.items() }
    assert False
    
def get_max_key(json):
    return fold_json(lambda x0, k, _: x0 if k is None else max(x0, k), 0, json)

def get_max_ident(json):
    return fold_json(lambda x0, _, v: x0 if v is None else max(x0, v), 0, json)

class Seq2JSONExample(macarico.Example):
    def __init__(self, tokens, out_json):
        super(Seq2JSONExample, self).__init__(tokens, out_json)
        #self.tokens = tokens
        #self.truth = out_json
        self.n_key = 1 + get_max_key(out_json)
        self.n_ident = 1 + get_max_ident(out_json)

class Seq2JSON(macarico.Env):
    NODE_ITEM, NODE_LIST, NODE_DICT = 0, 1, 2
    
    def __init__(self, ex, n_key, n_ident, max_depth=40, max_length=5, T=1000):
        macarico.Env.__init__(self, n_key + n_ident + 3 + 2, T, ex)
        self.n_key = n_key
        self.n_ident = n_ident
        self.max_depth = max_depth
        self.max_length = max_length
        self.Y = ex.Y
        self.X = ex.X

        assert(ex.X[-1] == EOS)
        self.X = ex.X
        
        self.actions_node_type = set([0,1,2])
        self.actions_stop = set([3,4])
        self.actions_string = set([i+5 for i in range(n_ident)])
        self.actions_key = set([i+5+n_ident for i in range(n_key)])
        
    def _run_episode(self, policy):
        self.policy = policy
        self.depth = 0
        self.count = 0
        #print('self.Y =', self.Y)
        self.out = self.generate_tree(self.Y)
        #print('out=', self.out)
        return self.out

    def _rewind(self):
        pass

    def output(self):
        return str(self.out)
    
    def generate_tree(self, truth):
        if self.depth > self.max_depth:
            return None

        self.count += 1
        if self.count >= self.horizon(): return None

        # predict node type
        self.gold_act = self.NODE_DICT if isinstance(truth, dict) else \
                        self.NODE_LIST if isinstance(truth, list) else \
                        self.NODE_ITEM if isinstance(truth, int) else \
                        None
        self.actions = self.actions_node_type
        #print('generate_tree:', self.depth, self.gold_act)
        node_type = self.policy(self)

        # generate corresponding type
        truth = None if (node_type != self.gold_act) else truth
        if node_type == self.NODE_ITEM:
            self.count += 1
            if self.count >= self.horizon(): return None

            self.actions = self.actions_string
            self.gold_act = None if truth is None else (truth + 5)
            return self.policy(self) - 5
        else:
            self.depth += 1
            res = self.generate_sequence(node_type == self.NODE_DICT, truth)
            self.depth -= 1
            return res
        
    def generate_sequence(self, is_dict, truth=None):
        res = {} if is_dict else []
        #print('generate_sequence:', is_dict, truth)
        true_keys = None if truth is None or not is_dict else sorted(truth.keys())
        
        for ii in range(self.max_length):
            self.count += 1
            if self.count >= self.horizon(): break
            
            self.actions = self.actions_stop
            self.gold_act = None if truth is None else \
                            3 if ii == len(truth) else \
                            4
            stop = self.policy(self)
            if stop == 3:
                break

            # if we're generating a dict, we need a key
            if is_dict:
                self.count += 1
                if self.count >= self.horizon(): break
                
                self.actions = self.actions_key
                self.gold_act = None if truth is None or ii >= len(true_keys) else \
                                (true_keys[ii] + 5 + self.n_ident)
                key = self.policy(self) - 5 - self.n_ident

            # for both lists and dicts we need a value
            ii_key = key if is_dict else ii
            true_item = None if truth is None else \
                        truth[ii] if (not is_dict and ii < len(truth)) else \
                        truth[true_keys[ii]] if (is_dict and ii < len(true_keys) and true_keys[ii] in truth) else \
                        None
            tree = self.generate_tree(true_item)

            if is_dict:
                res[key] = tree
            else:
                res.append(tree)

        return res

        
class JSONTreeFollower(macarico.Reference):
    def __init__(self):
        macarico.Reference.__init__(self)

    def __call__(self, state):
        assert state.gold_act is not None
        assert state.gold_act in state.actions, \
            str((state.gold_act, state.actions))
        #print('ref', state.gold_act, state.actions)
        return state.gold_act

    
class TreeEditDistance(macarico.Loss):
    corpus_level = False
    
    def __init__(self):
        super(TreeEditDistance, self).__init__('ted')
    
    def evaluate(self, example):
        if example.Y is None: return 999
        t_true = self.tree_to_zss(example.Y)
        t_pred = self.tree_to_zss(example.Yhat)
        return zss.simple_distance(t_true, t_pred)

    def tree_to_zss(self, t):
        if isinstance(t, int):
            return zss.Node(str(t))
        elif isinstance(t, list):
            node = zss.Node('**LIST**')
            for c in t:
                node.addkid(self.tree_to_zss(c))
            return node
        elif isinstance(t, dict):
            node = zss.Node('**DICT**')
            for k in sorted(t.keys()):
                child = zss.Node(k, [self.tree_to_zss(t[k])])
                node.addkid(child)
            return node
        elif t is None:
            return zss.Node('**NONE**')
        assert False, "don't know what to do with %s" % t

        
