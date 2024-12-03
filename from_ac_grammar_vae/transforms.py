import torch
from torchvision.transforms import Compose
import nltk


class MathTokenEmbedding:

    def __init__(self, alphabet, padding_token=" "):

        self.token_to_idx = { a: idx + 1 for idx, a in enumerate(alphabet)}
        self.idx_to_token = { idx + 1 : a for idx, a in enumerate(alphabet)}

        self.token_to_idx[padding_token] = 0
        self.idx_to_token[0] = padding_token

    def embed(self, tokens): #embed token to index
        return list(map(lambda t: self.token_to_idx[t], tokens))

    def decode(self, embeddings, pretty_print=False): #

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()

        decoded = list(map(lambda e: self.idx_to_token[e], embeddings))

        if pretty_print:
            return " ".join(decoded).strip() #process to get the whole expression

        return decoded

    def __call__(self, x):
        return self.embed(x)


class ToTensor:

    def __init__(self, dtype):
        self.dtype =dtype

    def __call__(self, x):
        return torch.as_tensor(x, dtype=self.dtype)


class PadSequencesToSameLength:

    def __call__(self, sequences):
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)


class GrammarParseTreeEmbedding:

    def __init__(self, context_free_grammar: nltk.grammar.CFG, pad_to_length=None):
        self._cfg : nltk.grammar.CFG = context_free_grammar
        self._parser = nltk.parse.ChartParser(self._cfg)

        self._prod_to_embedding = { None: 0 }
        self._embedding_to_prod = { 0: None }

        self._pad_to_length = pad_to_length

        for i, prod in enumerate(self._cfg.productions()):
            simple_production = nltk.Production(prod.lhs(), prod.rhs())
            self._prod_to_embedding[simple_production] = (i + 1)
            self._embedding_to_prod[i + 1] = simple_production

        self.non_terminals = set(map(lambda p: p.lhs(), self._cfg.productions()))
        self.start_symbol = self._cfg.start()

        self._length = len(self._cfg.productions()) + 1

    def __len__(self):
        return self._length

    def embed_all_productions_with(self, lhs):
        productions = self._cfg.productions(lhs=lhs)
        embeddings = map(lambda p: self._prod_to_embedding[nltk.Production(p.lhs(), p.rhs())], productions)
        return list(embeddings)

    def embed(self, expression):

        if isinstance(expression, torch.Tensor):
            expression = torch.split(expression, split_size_or_sections=1)

        if isinstance(expression, str):
            expression = [expression]

        if isinstance(expression, list):
            expression = [expression]

        # parse the expression with the CFG
        productions = [next(self._parser.parse(e)).productions() for e in expression]

        # look up the production indices and encode as one hot
        indices = [torch.as_tensor([self._prod_to_embedding[prod] for prod in seq], dtype=torch.int64) for seq in productions]

        # pad the sequences with NOOPs to the desired length
        if self._pad_to_length is not None:
            indices = [torch.nn.functional.pad(seq, pad=(0, self._pad_to_length - seq.shape[0]), mode="constant", value=self._prod_to_embedding[None]) for seq in indices]

        return torch.squeeze(torch.stack(indices, dim=0))

    def decode(self, embedding):
        return self._embedding_to_prod[embedding]

    def __call__(self, x):
        return self.embed(x)