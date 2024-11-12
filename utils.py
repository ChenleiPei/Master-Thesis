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
            word_id = word_tensor  # 这里直接使用整数值，不调用 .item()
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
from sympy import cos, exp, symbols
from functools import wraps

def convert_and_evaluate_expression(expression_str, x_value=None):
    """
    将字符串形式的数学表达式转换为可以被 SymPy 识别和计算的表达式。

    参数：
    - expression_str (str): 字符串形式的数学表达式。
    - x_value (float, optional): 用于替换符号 x 的值。

    返回：
    - sympy_expr (sympy.Expr): 转换后的 SymPy 表达式。
    - evaluated_value (float, optional): 如果提供了 x 的值，则返回计算结果。
    """
    expression = expression_str.replace("<start>", "").strip()

    expression = re.sub(r'(?<=\S)([\+\-\*/()])', r' \1 ', expression)  # 在符号前添加空格
    expression = re.sub(r'([\+\-\*/()])(?=\S)', r' \1 ', expression)  # 在符号后添加空格
    expression = re.sub(r'\s+', ' ', expression).strip()  # 去除多余空格

    expression = expression_str.replace("<start>", "").strip()
    #/ as devide
    #expression = re.sub(r'/', ' / ', expression)
    #* as multiply
    #expression = re.sub(r'\*', ' * ', expression)
    #( as (
    #expression = re.sub(r'\(', ' ( ', expression)
    #) as )
    #expression = re.sub(r'\)', ' ) ', expression)
    expression = re.sub(r'\bcos\b', 'cos', expression)
    expression = re.sub(r'\bexp\b', 'exp', expression)


    x = symbols('x')


    try:
        sympy_expr = sp.parse_expr(expression)
        print(f"SymPy Expression: {sympy_expr}")


        evaluated_value = None
        if x_value is not None:
            evaluated_value = sympy_expr.subs(x, x_value).evalf()
            print(f"Evaluated value (x={x_value}): {evaluated_value}")

        return convert_and_evaluate_expression, (sympy_expr, evaluated_value)

    except sp.SympifyError:
        print("Error: The expression could not be parsed.")
        return convert_and_evaluate_expression, (None, None)

# 示例使用
expression = "<start> 3 / cos ( 3 / x * 3 ) / exp ( 2 * 1 * 2 / 3 ) / 1 + 3"
convert_and_evaluate_expression(expression, x_value=2)