import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        num_nodes = config.get("num_nodes")
        mid_dim = config.get("hidden_dim")
        pts_dim = 3
        action_dim = 7

        self.dlo = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(num_nodes * pts_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )

        self.act = nn.Sequential(
            nn.Linear(action_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, mid_dim // 2),
            nn.ReLU(),
        )

        self.state_action = nn.Sequential(nn.Linear(2 * mid_dim, mid_dim), nn.ReLU())

        self.pred = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, num_nodes * pts_dim),
            nn.Unflatten(-1, (num_nodes, pts_dim)),
        )

    def forward(self, dlo, action0, action1):
        x_s = self.dlo(dlo)
        x_a0 = self.act(action0)
        x_a1 = self.act(action1)

        x = self.state_action(torch.concat([x_s, x_a0, x_a1], dim=-1))
        return self.pred(x) + dlo
