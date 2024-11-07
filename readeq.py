import re

def read_equations(file_path):
    """Read equations from a file, each line an equation"""
    with open(file_path, 'r') as file:
        equations = [line.strip() for line in file if line.strip()]
    return equations

def build_vocab(equations):
    """Build a vocabulary from a list of equations"""
    token_pattern = r"[a-zA-Z]+|\d+|[\+\-\*/\(\)]"
    vocab = set()
    for equation in equations:
        tokens = re.findall(token_pattern, equation)
        vocab.update(tokens)
    vocab = {token: idx for idx, token in enumerate(vocab, start=1)}
    vocab['<start>'] = len(vocab) + 1
    vocab['<end>'] = len(vocab) + 1
    vocab['<pad>'] = 0
    return vocab

def tokenize_and_generate_target(equations, vocab):
    """Convert equations to sequences of indices, adding special symbols, and generate targets"""
    token_pattern = r"[a-zA-Z]+|\d+|[\+\-\*/\(\)]"
    input_sequences = []
    target_sequences = []
    sequence_lengths = []
    max_length = 0
    for equation in equations:
        tokens = ['<start>'] + re.findall(token_pattern, equation)
        input_indices = [vocab[token] for token in tokens]
        target_indices = input_indices[1:] + [vocab['<end>']]
        input_sequences.append(input_indices)
        target_sequences.append(target_indices)
        sequence_lengths.append(len(input_indices))
        max_length = max(max_length, len(input_indices))

    padded_inputs = [seq + [vocab['<pad>']] * (max_length - len(seq)) for seq in input_sequences]
    padded_targets = [seq + [vocab['<pad>']] * (max_length - len(seq)) for seq in target_sequences]
    return padded_inputs, padded_targets, sequence_lengths

def process_equations(file_path):
    """Read equations from file, convert to symbol index sequences, generate targets, and calculate vocab size"""
    equations = read_equations(file_path)
    vocab = build_vocab(equations)
    padded_inputs, padded_targets, sequence_lengths = tokenize_and_generate_target(equations, vocab)
    alphabet_size = len(vocab) - 1  # Exclude <pad> from the alphabet size
    return alphabet_size, padded_inputs, padded_targets, sequence_lengths, vocab

# Use the function to process equations from a file and generate targets
#file_path = 'equations.txt'  # Replace with the actual file path
#alphabet_size, input_sequences, target_sequences, lengths, vocab = process_equations(file_path)


#print("Alphabet size:", alphabet_size)
#print("Input sequences:", input_sequences)
#print("Target sequences:", target_sequences)
#print("Lengths of each sequence before padding:", lengths)
