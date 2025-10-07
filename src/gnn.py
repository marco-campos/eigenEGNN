import json

import torch
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, LeakyReLU, Linear, Module, ModuleList, ReLU,
                      Sequential)
from torch_geometric.nn import (MLP, GINConv, GraphConv, ResGatedGraphConv,
                                global_add_pool, global_mean_pool)
from torch_geometric.nn.norm import BatchNorm


class GNNConfig:

    def __init__(
        self,
        name,
        input_channels,
        output_channels,
        hidden_channels=[64, 128, 256],
        MLP_hidden_channels=256,
        MLP_num_layers=10,
        per_layer_pooling=False,
        use_mlp=False,
        use_dropout=False,
        use_norm=False,
        use_area=False,
        use_dist=False,
        use_virtual_node=False,
    ):
        self.name = name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.MLP_hidden_channels = MLP_hidden_channels
        self.MLP_num_layers = MLP_num_layers
        self.per_layer_pooling = per_layer_pooling
        self.use_mlp = use_mlp
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.use_area = use_area
        self.use_dist = use_dist
        self.use_virtual_node = use_virtual_node

    def __str__(self):
        return f"GNNConfig(name={self.name}, input_channels={self.input_channels}, output_channels={self.output_channels}, hidden_channels={self.hidden_channels}, MLP_hidden_channels={self.MLP_hidden_channels}, MLP_num_layers={self.MLP_num_layers}), per_layer_pooling={self.per_layer_pooling}, use_mlp={self.use_mlp}, use_dropout={self.use_dropout}, use_norm={self.use_norm}, use_area={self.use_area}, use_dist={self.use_dist}, use_virtual_node={self.use_virtual_node}"

    def save_to_json(self, path):
        import json

        with open(path, "w") as file:
            json.dump(self.__dict__, file)  # save to json file

    @classmethod
    def load_from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
            return cls(**data)


class GNNLayer(Module):
    """
    GNN layer with GraphConv, LeakyReLU and Linear layers
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 use_mlp=False,
                 use_dropout=False):
        super().__init__()
        self.conv = GraphConv(in_channels, hidden_channels)
        self.relu = LeakyReLU(inplace=True)
        self.use_dropout = use_dropout
        if use_mlp:
            self.linear = MLP(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=3,
            )
        else:
            self.linear = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x)
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x


class GNN(Module):
    """
    GNN model with multiple GNN layers and an MLP layer
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels=[64, 128, 256],
        MLP_hidden_channels=256,
        MLP_num_layers=10,
        per_layer_pooling=False,
        use_mlp=False,
        use_dropout=False,
    ):
        super().__init__()
        torch.manual_seed(0xDEADBEEF)
        self.config = GNNConfig(
            input_channels,
            output_channels,
            hidden_channels,
            MLP_hidden_channels,
            MLP_num_layers,
            per_layer_pooling,
            use_mlp,
        )
        layers = []
        for hc in hidden_channels:
            layers.append(
                GNNLayer(input_channels,
                         hc,
                         use_mlp=use_mlp,
                         use_dropout=use_dropout))
            input_channels = hc

        self.layers = ModuleList(layers)
        self.mlp = MLP(
            in_channels=hidden_channels[-1],
            hidden_channels=MLP_hidden_channels,
            out_channels=output_channels,
            num_layers=MLP_num_layers,
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        for l in self.layers:
            x = l(x, edge_index, edge_attr)
            if self.config.per_layer_pooling:
                x = global_mean_pool(x, batch)

        if not self.config.per_layer_pooling:
            x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x

    @classmethod
    def load_from_config(cls, config):
        assert config.name == "GNN", "Invalid config name. Expecting 'GNN'"
        return cls(
            config.input_channels,
            config.output_channels,
            config.hidden_channels,
            config.MLP_hidden_channels,
            config.MLP_num_layers,
            config.per_layer_pooling,
            config.use_mlp,
            config.use_dropout,
        )


class SkipGNN(Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 hidden_channels=[64, 128, 256, 128, 128]):
        super().__init__()
        torch.manual_seed(0xDEADBEEF)
        mid = (len(hidden_channels) + 1) // 2
        layers = []
        for i in range(len(hidden_channels)):
            hc = hidden_channels[i]
            if i >= mid:
                input_channels += hidden_channels[len(hidden_channels) - 1 - i]
            layers.append(GNNLayer(input_channels, hc))
            input_channels = hc

        self.layers = ModuleList(layers)
        self.mlp = MLP(
            in_channels=hidden_channels[-1],
            hidden_channels=256,
            out_channels=output_channels,
            num_layers=10,
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        data = [x]
        mid = (len(self.layers) + 1) // 2
        for i in range(len(self.layers)):
            if i >= mid:
                x = torch.cat((x, data[len(self.layers) - 1 - i]), dim=1)
            x = self.layers[i](x, edge_index)
            data.append(x)

        x1 = global_mean_pool(x, batch)

        x2 = F.dropout(x1, p=0.5, training=self.training)
        x3 = self.mlp(x2)
        return x3


class QuadraticGNNLayer(Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = ResGatedGraphConv(in_channels, hidden_channels, act=ReLU())
        self.relu = LeakyReLU(inplace=True)
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x1 = self.conv(x, edge_index)
        x2 = self.relu(x1)
        x3 = self.lin(x2)
        return x3


class QuadraticGNN(Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 hidden_channels=[64, 128, 256]):
        super().__init__()
        torch.manual_seed(0xDEADBEEF)
        layers = []
        for hc in hidden_channels:
            layers.append(QuadraticGNNLayer(input_channels, hc))
            input_channels = hc

        self.layers = ModuleList(layers)
        self.mlp = MLP(
            in_channels=hidden_channels[-1],
            hidden_channels=64,
            out_channels=output_channels,
            num_layers=5,
        )

    def forward(self, x, edge_index, batch):
        x0 = x
        for l in self.layers:
            x0 = l(x0, edge_index)

        x1 = global_mean_pool(x0, batch)

        x2 = F.dropout(x1, p=0.5, training=self.training)
        x3 = self.mlp(x2)
        return x3


class BatchNormGNNLayer(Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        def xavier_init(mod):
            if type(mod) == Linear:
                torch.nn.init.xavier_uniform_(mod.weight)

        self.conv = GraphConv(in_channels, hidden_channels)
        self.conv.apply(xavier_init)
        self.relu1 = LeakyReLU(inplace=True)
        torch.nn.init.xavier_uniform_(
            self.relu1, torch.nn.init.calculate_gain("leaky_relu"))
        self.lin = Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.lin.weight)  # gain = 1
        self.gn = BatchNorm(hidden_channels)
        self.relu2 = LeakyReLU(inplace=True)
        torch.nn.init.xavier_uniform_(
            self.relu2.weight, torch.nn.init.calculate_gain("leaky_relu"))

    def forward(self, x, edge_index, batch, edge_attr=None):
        x1 = self.conv(x, edge_index, edge_attr)
        x2 = self.relu1(x1)
        x3 = self.lin(x2)
        x4 = self.gn(x3)
        x5 = self.relu2(x4)
        return x5


class BatchNormGNN(Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels=[64, 128, 256],
        MLP_hidden_channels=64,
        MLP_num_layers=5,
    ):
        super().__init__()
        torch.manual_seed(0xDEADBEEF)
        layers = []
        for hc in hidden_channels:
            layers.append(BatchNormGNNLayer(input_channels, hc))
            input_channels = hc

        self.layers = ModuleList(layers)
        self.mlp = MLP(
            in_channels=hidden_channels[-1],
            hidden_channels=MLP_hidden_channels,
            out_channels=output_channels,
            num_layers=MLP_num_layers,
        )

        def xavier_init(mod):
            if type(mod) == Linear:
                torch.nn.init.xavier_uniform_(mod.weight)

        self.mlp.apply(xavier_init)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x0 = x
        for l in self.layers:
            x0 = l(x0, edge_index, edge_attr)

        x1 = global_mean_pool(x0, batch)

        x2 = F.dropout(x1, p=0.5, training=self.training)
        x3 = self.mlp(x2)
        return x3


class GIN(Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels=[64, 128, 256],
        MLP_hidden_channels=64,
        MLP_num_layers=5,
    ):
        super().__init__()
        torch.manual_seed(0xDEADBEEF)
        layers = []
        for hc in hidden_channels:
            layers.append(
                GINConv(
                    Sequential(
                        Linear(input_channels, hc),
                        BatchNorm1d(hc),
                        ReLU(),
                        Linear(hc, hc),
                        ReLU(),
                    )))
            input_channels = hc

        self.layers = ModuleList(layers)
        self.mlp = MLP(
            in_channels=sum(hidden_channels),
            hidden_channels=MLP_hidden_channels,
            out_channels=output_channels,
            num_layers=MLP_num_layers,
        )

    def forward(self, x, edge_index, batch):
        y = [x]
        for l in self.layers:
            y.append(l(y[-1], edge_index))

        y = [global_add_pool(h, batch) for h in y[1:]]
        h0 = torch.cat(y, dim=1)
        h1 = F.dropout(h0, p=0.5, training=self.training)
        h2 = self.mlp(h1)
        return h2
