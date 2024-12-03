import torch
from LSTM_VAE_Model import LSTMVAE  # 假设模型类存放在 LSTM_VAE_Model.py 文件中
import argparse
import json
import torch
import torch.nn.functional as F


'''def decode_latent_point(model, latent_point, max_length, num_layers):
    """
    input: latent_point: torch.Tensor

    Args:
        model
        latent_point (torch.Tensor)
        max_length (int)

    Returns:
        expression (str)
    """
    # use the model for evaluation
    model.eval()
    with torch.no_grad():
        # latent_point = torch.tensor(latent_point, dtype=torch.float)
        latent_point = latent_point.unsqueeze(0)
        print("latent_point:",latent_point)
        hidden = model.decoder.latent_to_hidden(latent_point).unsqueeze(0)
        #deal with the lstm with num_layers=3
        hidden = hidden.repeat(num_layers, 1, 1)  # [num_layers, batch_size, hidden_size]
        #print("hidden:",hidden)
        cell = torch.zeros_like(hidden)  # 初始化 cell 状态为全零


        start_token = torch.tensor([[model.vocab['<start>']]], dtype=torch.long)
        embedded = model.decoder.embedding(start_token)


        outputs = []
        input_token = start_token

        for _ in range(max_length):

            embedded = model.decoder.embedding(input_token)
            output, (hidden, cell) = model.decoder.lstm(embedded, (hidden, cell))
            logits = model.decoder.fc(output)

            _, next_token = torch.max(logits, dim=-1)
            print("next_token:",next_token)

            outputs.append(next_token.item())

            if next_token.item() == model.vocab['<end>']:
                break

            input_token = next_token

        # transform the indices to expression
        outputs_tensor = torch.tensor(outputs, dtype=torch.long).view(1, -1)  # 保证它是一个 [1, seq_length] 张量

        # reconstruct the expression
        expression = model.reconstruct_expression(outputs_tensor, from_indices=True)

    return expression'''

'''def decode_latent_point(self, z, max_length, start_token_idx):
    # Map latent vector to hidden state
    z = z.unsqueeze(0)
    hidden = F.leaky_relu(self.latent_to_hidden(z)).unsqueeze(0)  # [1, batch_size, hidden_size]
    hidden = hidden.repeat(self.num_layers, 1, 1)  # Repeat for number of layers
    cell = torch.zeros_like(hidden)  # Initialize cell state with zeros

    # Initialize input sequence with <start> token
    batch_size = z.size(0)
    input_tokens = torch.full((batch_size, max_length), start_token_idx,
                                  dtype=torch.long)  # [batch_size, max_length]

    # Embed the entire input sequence
    embedded = self.embedding(input_tokens)  # [batch_size, max_length, embedding_size]

    # Pass the entire embedded sequence to LSTM
    lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # [batch_size, max_length, hidden_size]

    # Compute logits for all time steps
    logits = self.fc(lstm_out)  # [batch_size, max_length, vocab_size]

    # Get predicted tokens (optional: during inference)
    _, predicted_tokens = torch.max(logits, dim=-1)  # [batch_size, max_length]

    return predicted_tokens'''


def generate_interpolated_points_by_ratio(point1, point2, ratio):
    # to interpolate between two points
    interpolated_point = []

    # get the number of dimensions
    m = len(point1)

    # calculate the interpolated point
    for j in range(m):
        interpolated_point.append(point1[j] + ratio * (point2[j] - point1[j]))

    return interpolated_point


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode a latent point to generate the corresponding expression')
    parser.add_argument('--latent_point1', type=float, nargs='+', help='The latent point to be decoded, e.g., 0.5 1.2 -0.3', default=([0.10498531, -0.12027141, -0.65576386,  0.7685863,   0.07033981, -0.6881697, 0.85206336, -0.02666553, -0.18981981, -0.73386943, -0.07424233, -0.36830372, -0.53602314,  0.9901771,   0.89405996,  1.9209125,   0.71712655, -0.9048712, -1.4975828,   2.8080719,   1.0375603,   2.208498,   -0.37762815,  0.556855, -0.42313066,  1.7674068,   0.60604113,  2.1719244,  -0.29784876, -0.49175242, 0.10851225, -1.6908733]))#1 55
    parser.add_argument('--latent_point2', type=float, nargs='+',
                        help='The latent point to be decoded, e.g., 0.5 1.2 -0.3', default=(
    [-0.34220207, - 0.662769,    0.2798426, - 0.1654436,   0.3992821, - 1.2537338,
     0.7365996, - 0.40368897,  0.19032498, - 1.3094976, - 0.14064935,  1.7107444,
     - 0.7554677,   0.77114445, - 0.02718343,  3.3172626, - 0.70655495, - 0.7867145,
     0.8025752, - 1.2766687,   0.72836465,  1.1930429,   0.5323528,  1.9302149,
     1.5163276,   0.09037408,  0.807963, - 0.69715774, - 0.28639823, - 0.9673198,
     - 0.5875725, - 1.4622681])) # 154

    parser.add_argument('--model_path', type=str, help='Path to the trained VAE model file', default='./LSTMVAE_bin/2024-Nov-20-17-15-39/model_final.pth')
    parser.add_argument('--vocab_path', type=str, help='Path to the vocab JSON file', default='./LSTMVAE_bin/2024-Nov-20-17-15-39/vocab.json')
    parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer', default=64)
    parser.add_argument('--hidden_size', type=int, help='Size of the hidden layer', default=256)
    parser.add_argument('--latent_size', type=int, help='Size of the latent space', default=32)
    parser.add_argument('--num_layers', type=int, help='Number of layers in the LSTM', default=3)
    parser.add_argument('--max_sequence_length', type=int, help='Maximum length of the generated expression', default=40)

    args = parser.parse_args()

    # load the vocabulary
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    # load the trained model
    vocab_size = len(vocab)
    model = LSTMVAE(vocab_size=vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                    latent_size=args.latent_size, vocab=vocab, max_length=args.max_sequence_length,
                    num_layers=args.num_layers)

    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()


    #latent_point1 = torch.tensor(args.latent_point1, dtype=torch.float)
    #latent_point1 = latent_point1.unsqueeze(0)
    #print("shape of latent_point:",latent_point1.shape)

    #latent_point2 = torch.tensor(args.latent_point2, dtype=torch.float)
    #latent_point2 = latent_point2.unsqueeze(0)

    #use the latent_point to generate the expression
    #point1 use latent_point1
    #point2 use latent_point2

    point1 = args.latent_point1
    point2 = args.latent_point2
    ratio = 0.4

    interpolated_point = generate_interpolated_points_by_ratio(point1, point2, ratio)
    interpolated_point = torch.tensor(interpolated_point, dtype=torch.float)
    interpolated_point = interpolated_point.unsqueeze(0)
    print('interpolated_point:',interpolated_point)

    #expression = decode_latent_point(model, latent_point, max_length=args.max_sequence_length)
    logits, index = model.decoder(interpolated_point, max_length=args.max_sequence_length, start_token_idx=vocab['<start>'])
    print(f"Generated expression: {index}")

    #transfer the index to expression
    expression = model.reconstruct_expression(index, from_indices=True)
    print(f"Generated expression: {expression}")

