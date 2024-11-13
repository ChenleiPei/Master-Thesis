import torch
from LSTM_VAE_Model import LSTMVAE  # 假设模型类存放在 LSTM_VAE_Model.py 文件中
import argparse
import json


import torch

import torch


def decode_latent_point(model, latent_point, max_length=40):
    """
    输入一个潜空间的点，通过解码器生成输出，并重建具体表达式。

    Args:
        model: 训练好的VAE模型。
        latent_point (torch.Tensor): 潜空间的点，维度为 [latent_size]。
        max_length (int): 要生成的序列的最大长度。

    Returns:
        expression (str): 通过解码器生成的具体表达式。
    """
    # 确保模型在评估模式
    model.eval()
    with torch.no_grad():
        # 将潜在向量映射到隐藏状态
        latent_point = latent_point.unsqueeze(0)  # 添加 batch 维度，变为 [1, latent_size]
        print("latent_point:",latent_point)
        hidden = model.decoder.latent_to_hidden(latent_point).unsqueeze(0)  # 生成解码器隐藏状态
        #print("hidden:",hidden)
        cell = torch.zeros_like(hidden)  # 初始化 cell 状态为全零

        # 构造一个开始标记作为输入
        start_token = torch.tensor([[model.vocab['<start>']]], dtype=torch.long)
        embedded = model.decoder.embedding(start_token)

        # 初始化解码过程
        outputs = []
        input_token = start_token

        for _ in range(max_length):
            # 前向传播解码
            embedded = model.decoder.embedding(input_token)
            output, (hidden, cell) = model.decoder.lstm(embedded, (hidden, cell))
            logits = model.decoder.fc(output)

            # 选择具有最大概率的标记
            _, next_token = torch.max(logits, dim=-1)
            print("next_token:",next_token)

            # 将解码结果保存
            outputs.append(next_token.item())

            # 如果生成了 <end> 标记，停止生成
            if next_token.item() == model.vocab['<end>']:
                break

            # 使用当前的输出作为下一个时间步的输入
            input_token = next_token

        # 将生成的索引序列转换为张量
        outputs_tensor = torch.tensor(outputs, dtype=torch.long).view(1, -1)  # 保证它是一个 [1, seq_length] 张量

        # 重建具体表达式
        expression = model.reconstruct_expression(outputs_tensor, from_indices=True)

    return expression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode a latent point to generate the corresponding expression')
    parser.add_argument('--latent_point', type=float, nargs='+', help='The latent point to be decoded, e.g., 0.5 1.2 -0.3', default=[1, 1])
    parser.add_argument('--model_path', type=str, help='Path to the trained VAE model file', default='./LSTMVAE_bin/2024-Nov-13-13-38-57/model_final.pth')
    parser.add_argument('--vocab_path', type=str, help='Path to the vocab JSON file', default='./LSTMVAE_bin/2024-Nov-13-13-38-57/vocab.json')
    args = parser.parse_args()

    # 加载词汇表
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    # 定义模型结构并加载权重
    model = LSTMVAE(vocab_size=len(vocab), embedding_size=256, hidden_size=256, latent_size=2, vocab=vocab)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    # 将输入的潜空间点转换为张量
    latent_point = torch.tensor(args.latent_point, dtype=torch.float)

    # 通过解码器生成表达式
    expression = decode_latent_point(model, latent_point)
    print(f"Generated expression: {expression}")