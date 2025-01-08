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

    '''def embed(self, expression):

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

        return torch.squeeze(torch.stack(indices, dim=0))'''

    def embed(self, expression):
        # 如果输入是 torch.Tensor，将其分解为字符串列表
        if isinstance(expression, torch.Tensor):
            expression = [e.item() if e.numel() == 1 else e.tolist() for e in torch.split(expression, 1)]

        # 如果输入是字符串，将其包装为列表
        if isinstance(expression, str):
            expression = [expression]

        # 确保表达式是一个列表
        if isinstance(expression, list) and isinstance(expression[0], str):
            # 分词处理
            expression = [e.split() for e in expression]  # 简单按空格分割
            # 如果分词逻辑更复杂，使用 nltk.tokenize.wordpunct_tokenize(e)

        # 解析表达式为生产规则序列
        productions = [
            next(self._parser.parse(e)).productions() for e in expression
        ]

        # 将生产规则映射到索引
        indices = [
            torch.as_tensor([self._prod_to_embedding[prod] for prod in seq], dtype=torch.int64)
            for seq in productions
        ]

        # 填充序列到固定长度（如果需要）
        if self._pad_to_length is not None:
            indices = [
                torch.nn.functional.pad(
                    seq, pad=(0, self._pad_to_length - seq.shape[0]),
                    mode="constant", value=self._prod_to_embedding[None]
                )
                for seq in indices
            ]

        # 返回嵌入张量
        return torch.squeeze(torch.stack(indices, dim=0))


    def decode(self, embedding):
        return self._embedding_to_prod[embedding]

    def print_index_to_rule_mapping(self):
        """
        Print the mapping from indices to grammar productions.
        """
        for index in sorted(self._embedding_to_prod.keys()):
            production = self._embedding_to_prod[index]
            print(f"Index {index}: {production}")

    def get_index_to_rule_mapping(self):
        """
        Return a dictionary that maps indices to grammar productions.
        """
        return {index: str(production) for index, production in self._embedding_to_prod.items()}

    def from_index_to_expressions(self, indices):
        """
        Decode a sequence of indices back to their corresponding expressions by constructing
        a complete parse tree and then generating the expression.

        Parameters:
            indices (list of int or torch.Tensor): The sequence of indices to decode.

        Returns:
            str: The generated expression from the decoded production rules.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()  # Convert from Tensor to list if necessary

        # Decode indices to productions
        try:
            productions = [self._embedding_to_prod[index] for index in indices if
                           index in self._embedding_to_prod and index != 0]
        except KeyError as e:
            raise ValueError(f"An index in the sequence does not correspond to a valid production: {e}")

        # Build the expression from productions
        if not productions:
            return ""

        # Start with the start symbol of the grammar
        expression = [self._cfg.start()]
        try:
            for prod in productions:
                for i, symbol in enumerate(expression):
                    if not isinstance(symbol, str):  # If symbol is non-terminal
                        if symbol == prod.lhs():
                            expression[i:i + 1] = prod.rhs()  # Replace it with the right-hand side of the production
                            break
        except Exception as e:
            raise Exception(f"Error in constructing the expression: {e}")

        # Join all parts to form a complete expression
        return ' '.join(str(x) for x in expression if isinstance(x, str))


    def __call__(self, x):
        return self.embed(x)