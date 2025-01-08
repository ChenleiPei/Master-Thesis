import random

import tqdm
from torch.utils.data import Dataset
from typing import List, Any
import nltk
import time
from nltk.parse.generate import generate

from ac_grammar_vae.data.alphabet import alphabet


class CFGEquationDataset(Dataset):

    def __init__(self, n_samples=1000, transform=None, random_seed=2024, max_grammar_productions=32, use_original_grammar=True) -> None:
        self.n_samples = n_samples
        self.transform = transform
        self.random_seed = random_seed
        self.max_grammar_productions = max_grammar_productions
        
        # Initialize the grammar for the equation generation
        self.pcfg = self.__initialize_grammar(use_original_grammar=use_original_grammar)
    
    def __initialize_grammar(self, use_original_grammar):

        if use_original_grammar:
            rules = """
            S -> S '+' T
            S -> S '*' T
            S -> S '/' T
            S -> T
            T -> '(' S ')'
            T -> 'sin' '(' S ')'
            T -> 'cos' '(' S ')'
            T -> 'exp' '(' S ')'
            T -> 'x'
            T -> '1'
            T -> '2'
            T -> '3'
            
            """

        else:
            rules = """
            S -> S '+' T
            S -> S '*' T
            S -> S '/' T
            S -> S '**' T
            S -> T
            T -> '(' S ')'
            T -> 'sin' '(' S ')'
            T -> 'cos' '(' S ')'
            T -> 'exp' '(' S ')'
            T -> 'log' '(' S ')'
            T -> 'sqrt' '(' S ')'
            T -> 'x1'
            T -> 'x2'
            T -> 'x3'
            T -> 'theta'
            """

        cfg = nltk.CFG.fromstring(rules)
        pcfg = nltk.induce_pcfg(cfg.start(), cfg.productions()) #induce_pcfg() is used to convert a CFG to a PCFG, depending on the probabilities of the rules
        return pcfg

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index) -> Any: #get the producted expressions with pcfg
        expr = None
        retry = 0 #retry times
        while expr is None:
            random_seed = int(time.time() * 1000) + index
            #expr = sample_pcfg(self.pcfg, random_seed=hash(self.random_seed) + hash(index) + hash(retry), max_production_count=self.max_grammar_productions) #use hash to generate random seed
            expr = sample_pcfg(self.pcfg, random_seed=int(time.time() * 1000) + index + retry,
                               max_production_count=self.max_grammar_productions)
            retry += 1

        if self.transform:
            expr = self.transform(expr)

        return expr

    @classmethod
    def get_pcfg(cls, use_original_grammar=True):
        # 返回完整的 PCFG 规则
        from nltk import PCFG
        pcfg_rules = """
        S -> S '+' T [0.25]
        S -> S '*' T [0.25]
        S -> S '/' T [0.25]
        S -> T [0.25]
        T -> '(' S ')' [0.125]
        T -> 'sin' '(' S ')' [0.125]
        T -> 'cos' '(' S ')' [0.125]
        T -> 'exp' '(' S ')' [0.125]
        T -> 'x' [0.125]
        T -> '1' [0.125]
        T -> '2' [0.125]
        T -> '3' [0.125]
        """
        return PCFG.fromstring(pcfg_rules)

    def save(self, filename):
        with open(filename, "w") as f:
            for idx in tqdm.tqdm(range(len(self))):
                f.write(" ".join(self[idx]))
                f.write("\n")


def sample_pcfg(pcfg: nltk.grammar.PCFG, max_production_count=15, random_seed=None):
    terminals = [pcfg.start()] #start symbol
    search_from_idx = 0
    productions_used = 0

    rand = random.Random()
    rand.seed(random_seed)

    # while it contains non-terminal
    while search_from_idx < len(terminals): #make sure all the non-terminal are replaced

        if productions_used > max_production_count:
            return None

        # filter production rules that can be applied
        prods = pcfg.productions(lhs=terminals.pop(search_from_idx))

        # randomly select a production (with assigned probs.)
        prod = rand.choice(prods)

        # apply the production
        [terminals.insert(search_from_idx, s) for s in reversed(prod.rhs())]
        productions_used += 1

        # find index of the first non-terminal
        idx = len(terminals)
        for i in range(search_from_idx, idx):
            if not isinstance(terminals[i], str):
                idx = i
                break
        
        # search next time from this index
        search_from_idx = idx

    return terminals
