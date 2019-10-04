import torch
import torch.nn as nn

from macarico.util import Varng


class Regressor(nn.Module):
    def __init__(self, dim, n_hid_layers=0, hidden_dim=15, loss_fn='squared'):
        nn.Module.__init__(self)
        if n_hid_layers > 0:
            self.model = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        else:
            self.model = nn.Sequential(nn.Linear(dim, 1))
        self.set_loss(loss_fn)

    def set_loss(self, loss_fn):
        assert loss_fn in ['squared', 'huber']
        self.loss_fn = nn.MSELoss(reduction='sum') if loss_fn == 'squared' else \
            nn.SmoothL1Loss(reduction='sum') if loss_fn == 'huber' else None

    def forward(self, inp):
        z = self.model(inp)
        return z

    def update(self, pred, feedback):
        return self.loss_fn(pred, Varng(feedback))


