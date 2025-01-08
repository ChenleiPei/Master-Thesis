import torch


class GrammarDecoderNetwork(torch.nn.Module):

    def __init__(self,
                 num_grammar_productions,
                 max_of_production_steps,
                 latent_dim=2,
                 gru_num_units=64,
                 gru_num_layers=3,
                 conv_filter_sizes=[],
                 conv_activation=torch.nn.ReLU(),
                 conv_applies_batch_norm=True,
                 dense_units=[],
                 dense_activation=torch.nn.ReLU()):
        super(GrammarDecoderNetwork, self).__init__()

        self.max_production_steps = max_of_production_steps
        self.gru_num_units = gru_num_units
        self.gru_num_layers = gru_num_layers
        self.pre_network = torch.nn.Sequential()

        # does it make sense to start with a batch-norm ?!
        self.pre_network.append(torch.nn.BatchNorm1d(num_features=latent_dim))

        # linear layer
        self.pre_network.append(torch.nn.Linear(in_features=latent_dim, out_features=latent_dim))

        # add GRU layers
        self.gru = torch.nn.GRU(
                input_size=latent_dim,
                hidden_size=gru_num_units,
                num_layers=gru_num_layers,
                batch_first=True
            )

        self.output_network = torch.nn.Sequential(
            torch.nn.Linear(gru_num_units, num_grammar_productions)
        )

    def forward(self, z, mask=None):

        with torch.no_grad():
            if z is None:
                z = torch.randn([1, self.latent_dim])  # 确保 z 是二维张量
            elif z.ndim == 1:
                z = z.unsqueeze(0)

        bs, embedding_size = z.shape

        # apply a linear layer first
        z_transformed = self.pre_network(z)

        # duplicate the embedding along the time-axis
        inputs = torch.unsqueeze(z_transformed, dim=1).expand(-1, self.max_production_steps, -1)

        # generate the initial state
        h0 = torch.zeros((self.gru_num_layers, bs, self.gru_num_units))

        # apply the GRU
        X, h = self.gru(inputs, h0)

        # time-distributed output layer
        logits = self.output_network(X)

        # mask is not None:
            #logits = torch.masked_fill(logits, mask=~mask, value=float("-inf"))
            #logits = torch.masked_fill(logits, mask=~mask.bool(), value=float("-inf"))

        return logits