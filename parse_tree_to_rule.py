import nltk
from nltk import CFG
from typing import List, Dict


class RuleMapper:
    """Class to map rules to indices and parse expressions."""
    def __init__(self, rules: str):
        """
        Initialize with a set of rules and map them to indices.
        Args:
            rules: The string of grammar rules.
        """
        self.cfg = CFG.fromstring(rules)
        self.rule_to_index = self._create_rule_index_map()



    def _create_rule_index_map(self) -> Dict[str, int]:
        """
        Create a mapping from rules to indices.
        Returns:
            A dictionary mapping rules to unique indices.
        """
        # Ensure consistent formatting for all rules
        return {self._format_rule(prod): idx for idx, prod in enumerate(self.cfg.productions())}

    def _format_rule(self, prod) -> str:
        """
        Standardize the format of a rule.
        Args:
            prod: A production rule.
        Returns:
            A standardized string representation of the rule.
        """
        lhs = str(prod.lhs())
        rhs = " ".join(map(str, prod.rhs()))
        return f"{lhs} -> {rhs}".strip()

    def parse_expression(self, expression: str) -> List[int]:
        """
        Parse the expression into a sequence of rule indices.
        Args:
            expression: The input expression as a string.
        Returns:
            A list of rule indices representing the parsing process.
        """
        parser = nltk.ChartParser(self.cfg)
        try:
            trees = list(parser.parse(expression.split()))
        except ValueError as e:
            raise ValueError(f"Error parsing expression: {expression}. {e}")

        if not trees:
            raise ValueError("The given expression cannot be parsed with the provided rules.")

        # Convert the parse tree to a sequence of rules
        return self._extract_rule_sequence(trees[0])  # Use the first parse tree

    def _extract_rule_sequence(self, tree: nltk.Tree) -> List[int]:
        """
        Extract the sequence of rule indices from a parse tree.
        Args:
            tree: The parse tree.
        Returns:
            A list of rule indices.
        """
        rule_sequence = []

        def traverse(subtree):
            if not isinstance(subtree, nltk.Tree):  # If terminal, skip
                return
            # Get the rule from the current subtree
            lhs = subtree.label()
            rhs = [child.label() if isinstance(child, nltk.Tree) else child for child in subtree]
            rule = f"{lhs} -> {' '.join(map(str, rhs))}".strip()
            formatted_rule = self._format_rule(nltk.grammar.Production(nltk.Nonterminal(lhs), rhs))
            if formatted_rule not in self.rule_to_index:
                raise KeyError(f"Rule not found in mapping: {formatted_rule}")
            rule_sequence.append(self.rule_to_index[formatted_rule])
            # Recursively traverse children
            for child in subtree:
                traverse(child)

        traverse(tree)
        return rule_sequence


if __name__ == "__main__":
    # Define grammar rules
    rules = """
        S -> S '+' T
        S -> S '*' T
        S -> T
        T -> '(' S ')'
        T -> 'x'
        T -> '2'
    """

    # Input expression
    input_expression = "x + 2"

    # Initialize the RuleMapper
    rule_mapper = RuleMapper(rules)

    # Parse the expression and print the rule sequence
    try:
        rule_sequence = rule_mapper.parse_expression(input_expression)
        print("Expression:", input_expression)
        print("Rule Sequence:", rule_sequence)
        print("Rule to Index Mapping:", rule_mapper.rule_to_index)
    except ValueError as e:
        print("Parsing Error:", e)
    except KeyError as e:
        print("Rule Mapping Error:", e)

