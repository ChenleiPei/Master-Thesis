def calculate_probability(expression, rules):
    # 计算每个非终结符号的产生规则的数量
    rule_counts = {}
    for rule in rules:
        lhs = rule.split('->')[0].strip()
        rule_counts[lhs] = rule_counts.get(lhs, 0) + 1

    # 递归地计算表达式的生成概率
    def parse_probability(expr):
        if expr.isdigit():  # 如果是数字
            return 1 / rule_counts['T']  # T -> '1', T -> '2', ...
        elif expr in ['x']:  # 如果是变量 x
            return 1 / rule_counts['T']
        elif expr.startswith('(') and expr.endswith(')'):
            return (1 / rule_counts['T']) * parse_probability(expr[1:-1])  # T -> '(' S ')'
        elif expr.startswith('sin(') and expr.endswith(')'):
            return (1 / rule_counts['T']) * parse_probability(expr[4:-1])  # T -> 'sin' '(' S ')'
        elif expr.startswith('cos(') and expr.endswith(')'):
            return (1 / rule_counts['T']) * parse_probability(expr[4:-1])
        elif expr.startswith('exp(') and expr.endswith(')'):
            return (1 / rule_counts['T']) * parse_probability(expr[4:-1])
        # 处理 S 规则
        index = max(expr.rfind('+'), expr.rfind('*'), expr.rfind('/'))
        if index > -1:
            left_prob = parse_probability(expr[:index])
            right_prob = parse_probability(expr[index + 1:])
            return (1 / rule_counts['S']) * left_prob * right_prob  # S -> S '+' T, ...
        return 1 / rule_counts['S']  # S -> T

    return parse_probability(expression)


# 语法规则
cfg_rules = [
    "S -> S '+' T",
    "S -> S '*' T",
    "S -> S '/' T",
    "S -> T",
    "T -> '(' S ')'",
    "T -> 'sin' '(' S ')'",
    "T -> 'cos' '(' S ')'",
    "T -> 'exp' '(' S ')'",
    "T -> 'x'",
    "T -> '1'",
    "T -> '2'",
    "T -> '3'"
]

# 示例调用
example_expression = "sin(x+1)"
probability = calculate_probability(example_expression, cfg_rules)
print(f"The probability of generating '{example_expression}' is {probability:.5f}")
