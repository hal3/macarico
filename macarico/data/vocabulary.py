from __future__ import division, generators, print_function

EOS = 0
OOV = 1

class Vocabulary(object):
    def __init__(self, lowercase=True, include_special=True):
        self.w2i = {}
        self.i2w = []
        self.lowercase = lowercase
        self.include_special = include_special
        if include_special:
            self.i2w = ['</s>', '<oov>']
            self.w2i = { w: i for i, w in enumerate(self.i2w) }
        self.frozen = False

    def __call__(self, item):
        if isinstance(item, int):
            return self.get_int(item)
        return self.get_word(item)

    def freeze(self):
        self.frozen = True
        
    def unfreeze(self):
        self.frozen = False

    def __len__(self):
        return len(self.i2w)
        
    def get_int(self, i):
        assert i >= 0 and i < len(self.i2w)
        return self.i2w[i]

    def get_word(self, s):
        if self.lowercase: s = s.lower()
        i = None
        if s in self.w2i:
            return self.w2i[s]
        elif self.frozen:
            return self.w2i['<oov>']
        else:
            i = len(self.i2w)
            self.w2i[s] = i
            self.i2w.append(s)
            return i

    def __contains__(self, item):
        if isinstance(item, int):
            return 0 <= item and item < len(self.i2w)
        return item in self.w2i

    def __getitem__(self, item): return self(item)
    
    def items(self):
        for i, w in enumerate(self.w2i):
            yield w, i

class NoVocabulary(Vocabulary):
    def __init__(self, n_items): self.n_items = n_items
    def __len__(self): return self.n_items
    def __call__(self, x): return x
            
    
