import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x): #将数据转换为Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def idx2word(idx, i2w, pad_idx):
    # 如果 idx 是一维列表，直接处理
    if isinstance(idx[0], int):
        sent_str = ""
        for word_id in idx:
            if word_id == pad_idx:
                break
            # 获取单词
            word = i2w.get(word_id, '<unk>')
            sent_str += word + " "
        return [sent_str.strip()]

    # 原来的逻辑
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_tensor in sent:
            word_id = word_tensor
            if word_id == pad_idx:
                break
            word = i2w.get(word_id, '<unk>')
            sent_str[i] += word + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str






def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name


import re
import sympy as sp
from sympy import symbols

def convert_expression(expression_str):
    """
    convert_expression function will convert a string expression to a SymPy expression.
    """
    # Remove "<start>" if it exists
    expression = expression_str.replace("<start>", "").strip()
    expression = re.sub(r'(?<=\S)([\+\-\*/()])', r' \1 ', expression)  # pre-add space
    expression = re.sub(r'([\+\-\*/()])(?=\S)', r' \1 ', expression)  # post-add space
    expression = re.sub(r'\s+', ' ', expression).strip()  # remove extra spaces
    expression = re.sub(r'\bcos\b', 'cos', expression)
    expression = re.sub(r'\bexp\b', 'exp', expression)

    try:
        # let sympy parse the expression
        with sp.evaluate(False): #let the evaluate be false
            sympy_expr = sp.parse_expr(expression, evaluate=False)
            #sympy_expr = sp.parse_expr(expression)
            print(f"Converted SymPy Expression: {sympy_expr}")
        print(f"Converted SymPy Expression: {sympy_expr}")
        return sympy_expr

    except sp.SympifyError:
        print("Error: The expression could not be parsed.")
        return None


import numpy as np
from sympy import symbols, sympify

def evaluate_expression(sympy_expr, x_range=None):
    """
    Evaluate a SymPy expression for a given range of x values or a single x value.
    """
    x = symbols('x')

    if sympy_expr is None:
        print("Error: Invalid SymPy expression.")
        return None

    try:
        # If x_range is a tuple (start, stop, step)
        if isinstance(x_range, (list, tuple)) and len(x_range) == 3:
            start, stop, step = x_range
            x_values = np.arange(start, stop, step)  # Use NumPy to create range
            results = [sympy_expr.subs(x, value).evalf() for value in x_values]
            print(f"Evaluated values for range {x_range}: {results}")
            return results

        # If x_range is a single number
        elif isinstance(x_range, (int, float)):
            result = sympy_expr.subs(x, x_range).evalf()
            print(f"Evaluated value (x={x_range}): {result}")
            return result

        else:
            print("Error: Invalid x_range input. It must be a number or a tuple (start, stop, step).")
            return None

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


# Example usage
#let the evaluate be false

def convert_expression2(expression_str):
    """
    Converts a string expression into a SymPy expression, removing unwanted tokens like "<start>".
    """
    # Remove "<start>" if it exists
    cleaned_expression = expression_str.replace("<start>", "").strip()
    try:
        sympy_expr = sympify(cleaned_expression, evaluate=False)
        print(f"Converted SymPy Expression: {sympy_expr}")
        return sympy_expr
    except Exception as e:
        print(f"Error converting expression: {e}")
        return None


#expression_str = "<start> x*x + 2*x + 1/1"
#sympy_expr = convert_expression2(expression_str)
#evaluate_expression(sympy_expr, x_range=(0, 10, 1))
