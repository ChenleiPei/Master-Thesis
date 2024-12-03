import nltk
import time
import random
from typing import List, Union


class TreeNode:
    """to represent a node in the tree"""
    def __init__(self, value: str):
        self.value = value
        self.children: List[TreeNode] = []  #list of children

    def add_child(self, child):
        self.children.append(child)

    def is_terminal(self):
        """if the node is terminal"""
        return not self.children

    def __repr__(self):
        """print"""
        if self.is_terminal():
            return self.value
        return f"{self.value}({', '.join(map(str, self.children))})"


def sample_pcfg_as_tree(pcfg: nltk.grammar.PCFG, max_production_count=15, random_seed=None):
    """
    use PCFG to generate a parse tree
    Args:
        pcfg
        max_production_count
        random_seed
    Returns:
        root
    """
    rand = random.Random()
    rand.seed(random_seed)

    # create the root node
    root = TreeNode(pcfg.start())
    productions_used = 0

    def expand_node(node, productions_used):

        if productions_used > max_production_count:
            return None, productions_used


        prods = pcfg.productions(lhs=node.value)
        if not prods:  # 如果没有生成规则，说明是终结符
            return node, productions_used

        # 随机选择一个生成规则
        prod = rand.choice(prods)
        productions_used += 1


        for symbol in prod.rhs():
            child = TreeNode(symbol)
            node.add_child(child)
            if not isinstance(symbol, str):  # 如果是非终结符，递归展开
                child, productions_used = expand_node(child, productions_used)

        return node, productions_used

    root, _ = expand_node(root, productions_used)
    return root


def tree_to_expression(node: TreeNode) -> str:
    """
    transfer the tree to expression
    Args:
        node: root node of the tree
    Returns:
        expression: the expression
    """
    if node.is_terminal():
        return node.value
    return "".join(tree_to_expression(child) for child in node.children)


# 测试代码
if __name__ == "__main__":
    # define the rules of CFG
    rules = """
        S -> S '+' T
        S -> S '*' T
        S -> T
        T -> '(' S ')'
        T -> 'x'
        T -> '2'
    """
    cfg = nltk.CFG.fromstring(rules)
    pcfg = nltk.induce_pcfg(cfg.start(), cfg.productions())


    random_seed = int(time.time())
    tree = sample_pcfg_as_tree(pcfg, max_production_count=10, random_seed=random_seed)
    print("parse tree:", tree)


    # transfer the tree to expression
    expression = tree_to_expression(tree)
    print("expression:", expression)
