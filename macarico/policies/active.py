from __future__ import division
import random
import dynet as dy
import numpy as np

from macarico import Policy

class CSActive(Policy):
    def __init__(self, base_policy, min_cost=0, max_cost=1, mellowness=0.1, range_c=0.5):
        self.base_policy = base_policy
        self.optimizer = None
        
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.mellowness = mellowness
        self.range_c = range_c
        self.t = 0
        self.cost_span = self.max_cost - self.min_cost
        self.num_query = 0
        self.num_skip = 0

        self.MAX_ITER = 20
        self.TOLERANCE = 1e-6

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    #############################
    ## passthrough to base policy
    #############################
        
    def __call__(self, state, deviate_to=None):
        b, z = self.cost_ranges(state)
        return self.base_policy(state, deviate_to)

    def sample(self, state):
        return self.base_policy.sample(state)

    def stochastic(self, state, temperature=1):
        return self.base_policy.stochastic(state, temperature)

    def stochastic_with_probability(self, state, temperature=1):
        return self.base_policy.stochastic_with_probability(state, temperature)
        
    def predict_costs(self, state, deviate_to=None):
        return self.base_policy.predict_costs(state, deviate_to)
    
    def forward(self, state, truth):
        return self.base_policy.forward(state, truth)
    
    def forward_partial_complete(self, pred_costs, truth, actions):
        #print truth
        assert all((self.min_cost <= t for t in truth))
        assert all((t <= self.max_cost for t in truth))
        if np.random.random() < 0.001:
            print self.num_query / max(1, self.num_query + self.num_skip)
        return self.base_policy.forward_partial_complete(pred_costs, truth, actions)


    ########################
    ## active learning stuff
    ########################
    def cost_ranges(self, state, pred_costs=None):
        if pred_costs is None:
            pred_costs = self.predict_costs(state)
        if isinstance(pred_costs, dy.Expression):
            pred_costs = pred_costs.npvalue()
            
        self.t += 1
        K = len(pred_costs)
        t_prev = max(self.t - 1, 1)
        min_max_cost = 1e100
        eta = self.range_c * self.cost_span / np.sqrt(self.t)
        delta = self.mellowness * np.log(K * t_prev) * (self.cost_span ** 2)
    
        ranges = []
        feats = self.base_policy.features(state).npvalue()
        #if self.t > 22000:
        #    from arsenal import ip; ip()
        sens = self.sensitivity(feats)
        for i in xrange(K):
            min_pred, max_pred, is_range_large = \
              self.find_cost_range(feats, pred_costs, i, delta, eta, sens)
            min_max_cost = min(min_max_cost, max_pred)
            ranges.append( (min_pred, max_pred, is_range_large) )
    
        n_overlapped = 0
        for (min_pred, max_pred, is_range_large) in ranges:
            is_range_overlapped = min_pred <= min_max_cost
            n_overlapped += is_range_overlapped

        query_any = False
        label_cost_ranges = []
        if True or n_overlapped > 1:
            for i, (min_pred, max_pred, is_range_large) in enumerate(ranges):
                is_range_overlapped = min_pred <= min_max_cost
                to_query = is_range_overlapped and is_range_large and n_overlapped > 1
                label_cost_ranges.append( (to_query, min_pred, max_pred) )
                query_any = query_any or to_query
                if to_query:
                    self.num_query += 1
                else:
                    self.num_skip += 1

        #if not query_any:
        #    from arsenal import ip; ip()
        return query_any, label_cost_ranges
    
    def compute_rate_decay(self, w):
        rate_decay = 1.
        # TODO: this really needs to take, eg, adagrad params into account
        return rate_decay

    def average_update(self):
        return 1. # if normalized this is different
        
    def get_pred_per_update(self, x):
        #x_min = 1.084202e-19
        #x2_min = x_min * x_min
        
        #grad_squared = loss.get_square_grad(pred, label) * imp_weight
        #grad_squared = (pred - label) * (pred - label) * imp_weight

        # TODO: pred_per_update_feature
        pred_per_update = x.dot(x)
        #for x, w in zip(features, weights):
            # right now, compute_rate_decay is just 1, so this is the sum of x^2
            #x2 = x * x
            #if x2 < x2_min:
            #    x = x_min if x > 0 else -x_min
            #x2 = x2_min
            #pred_per_update += x2 * self.compute_rate_decay(w)

        pred_per_update *= self.average_update()
        return pred_per_update
        
    def sensitivity(self, x):
        # note: "stateless" is true
        return self.get_scale() * self.get_pred_per_update(x)

    def get_scale(self):
        return self.optimizer.learning_rate / np.sqrt(max(1, self.t))
    
    def binary_search(self, fhat, delta, sens):
        fhat2 = fhat*fhat
        
        maxw = min(fhat/sens, 1e20)
        if maxw*fhat2 <= delta:
            return maxw
    
        l = 0
        u = maxw
        w = None
        v = None
        for _ in xrange(self.MAX_ITER):
            w = (u + l) / 2
            v = w * (fhat2 - (fhat-sens*w)*(fhat-sens*w)) - delta
            l, u = (l, w) if v > 0 else (w, u)
            if abs(v) < self.TOLERANCE or u-l <= self.TOLERANCE:
                break
            
        return l
    
    def find_cost_range(self, x, pred_costs, i, delta, eta, sens):
        #if np.random.random < 0.1: print sens
        if np.isnan(sens) or np.isinf(sens):
            return self.min_cost, self.max_cost, True
    
        max_pred = min(self.max_cost,
                       pred_costs[i] + \
                       sens * self.binary_search(self.max_cost - pred_costs[i], delta, sens))
        min_pred = max(self.min_cost,
                       pred_costs[i] - \
                       sens * self.binary_search(pred_costs[i] - self.min_cost, delta, sens))
        is_range_large = max_pred - min_pred > eta
    
        return min_pred, max_pred, is_range_large
    

