import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import SentenceVAE
from utils import to_var
from readeq import process_equations


import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import SentenceVAE  # 确保导入你的模型类
from utils import to_var
from readeq import process_equations

def main(args):
    # 加载词汇表
    vocab_path = os.path.join(args.model_dir, 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # 加载模型
    model_path = os.path.join(args.model_dir, 'model_final.pth')
    model = torch.load(model_path)  # 直接加载整个模型
    model.eval()  # 设置为评估模式

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU.")

    # 数据加载和处理
    vocab_size, input_sequences, target, lengths, _ = process_equations(args.data_file)
    input_tensor = torch.tensor(input_sequences, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    # 创建 DataLoader
    dataset = TensorDataset(input_tensor, lengths_tensor)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 保存 μ 和 σ
    mu_list = []
    sigma_list = []

    # 前向传播
    with torch.no_grad():  # 禁用梯度计算
        for batch_input, batch_lengths in data_loader:
            batch_input = to_var(batch_input)
            batch_lengths = to_var(batch_lengths)

            # 前向传播，获取 μ 和 log_var
            mu, log_var = model.encoder(batch_input, batch_lengths)
            sigma = torch.exp(0.5 * log_var)  # 计算 σ

            # 保存 μ 和 σ
            mu_list.append(mu.cpu().numpy())
            sigma_list.append(sigma.cpu().numpy())

    # 将 μ 和 σ 拼接为完整的数组
    mu_array = np.concatenate(mu_list, axis=0)
    sigma_array = np.concatenate(sigma_list, axis=0)

    # 保存 μ 和 σ 到文件
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    mu_path = os.path.join(output_dir, 'mu.npy')
    sigma_path = os.path.join(output_dir, 'sigma.npy')
    np.save(mu_path, mu_array)
    np.save(sigma_path, sigma_array)

    print(f"μ saved to {mu_path}")
    print(f"σ saved to {sigma_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,  default='equations_30_isotherm.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--model_dir', type=str, default='LSTMVAE_bin/2024-Dec-04-20-23-51',
                        help='Directory containing the trained model and vocab files')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save μ and σ')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    args = parser.parse_args()

    main(args)