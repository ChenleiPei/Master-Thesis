import torch


class CharacterStringEncoderNetwork(torch.nn.Module):

    def __init__(self, context_free_grammar):
        self._cfg = context_free_grammar

    def __call__(self, X):
        z = None

        return z