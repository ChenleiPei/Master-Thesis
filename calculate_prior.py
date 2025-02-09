import re

def calculate_probability(expression, rules):
    # 解析规则并构建规则映射和概率映射
    rule_map = {}
    probabilities = {}

    for rule in rules:
        lhs, rhs = rule.split('->')
        lhs = lhs.strip()
        options = rhs.strip().split('|')
        rule_map[lhs] = [option.strip().split() for option in options]
        probabilities[lhs] = 1 / len(options)

    # 递归函数来计算表达式的生成概率
    def parse_probability(expr, symbol):
        # 初始化总概率为0
        total_prob = 0

        # 对于每个产生式
        for production in rule_map.get(symbol, []):
            # 创建匹配模式
            regex_parts = []
            for part in production:
                if part.startswith("'") and part.endswith("'"):  # 处理字面量符号
                    regex_parts.append(re.escape(part.strip("'")))
                    print(regex_parts)
                elif part.isupper():  # 处理非终结符
                    regex_parts.append(f"({part})")
            pattern = ''.join(regex_parts)
            match = re.fullmatch(pattern, expr)

            # 如果匹配成功，计算子表达式的概率
            if match:
                prob = probabilities[symbol]  # 当前产生式的概率
                for idx, sub_symbol in enumerate(production):
                    if sub_symbol.isupper():  # 是非终结符
                        sub_expr = match.group(idx+1)
                        sub_prob = parse_probability(sub_expr, sub_symbol)
                        prob *= sub_prob
                total_prob += prob

        return total_prob

    return parse_probability(expression, 'S')

# 定义规则
cfg_rules = [
    "S -> S '+' T | T",
    "T -> '(' S ')' | 'sin' '(' S ')' | 'cos' '(' S ')' | 'exp' '(' S ')' | 'x' | '1' | '2' | '3'"
]

# 测试表达式
example_expression = "sin(x+1)"
probability = calculate_probability(example_expression, cfg_rules)
print(f"The probability of generating '{example_expression}' from 'S' is {probability:.5f}")


