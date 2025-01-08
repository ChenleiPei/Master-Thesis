import torch


class GrammarEncoderNetwork(torch.nn.Module):

    def __init__(self,
                 num_grammar_productions,
                 max_of_production_steps,
                 latent_dim = 2,
                 conv_num_filters=[],
                 conv_filter_sizes=[],
                 conv_activation=torch.nn.ReLU(),
                 conv_applies_batch_norm=True,
                 dense_units=[],
                 dense_activation=torch.nn.ReLU()):
        
        super(GrammarEncoderNetwork, self).__init__()
        
        self.latent_dim = latent_dim

        # build the convolutional network
        self.conv_network = torch.nn.Sequential()
        in_channels = num_grammar_productions
        for i, (num_filters, filter_size) in enumerate(zip(conv_num_filters, conv_filter_sizes)):
            self.conv_network.add_module(f"conv1d_{i}", torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding='same',
                padding_mode='zeros'
            ))
            self.conv_network.add_module(f"actv_{i}", conv_activation)

            if conv_applies_batch_norm:
                self.conv_network.add_module(f"bnorm_{i}", torch.nn.BatchNorm1d(num_features=num_filters))

            in_channels = num_filters

        # flatten the outputs for the dense layers
        self.flatten_production_conv = torch.nn.Flatten()

        # build the dense network
        self.dense_network = torch.nn.Sequential()
        in_features = max_of_production_steps * in_channels
        for i, n_units in enumerate(dense_units):
            self.dense_network.add_module(f"dense_{i}", torch.nn.Linear(in_features=in_features, out_features=n_units))
            self.dense_network.add_module(f"actv_{i}", dense_activation)
            in_features = n_units

        # add an output layer
        self.dense_network.add_module(
            name="out",
            module=torch.nn.Linear(in_features=in_features, out_features=2*self.latent_dim)
        )

    def forward(self, X):

        X = X.permute(0, 2, 1)

        # CNN encoder
        z = self.conv_network(X)
        z = self.flatten_production_conv(z)
        z = self.dense_network(z)

        # determine the parameters of the variational distribution
        mean, log_std = torch.split(z, dim=-1, split_size_or_sections=self.latent_dim)
        print("shape of mean: ", mean.shape)
        print("shape of log_std: ", log_std.shape)

        return mean, log_std #may used to define the mean and standard deviation in the latent space?