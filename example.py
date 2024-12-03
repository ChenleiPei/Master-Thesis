import torch
from torch.utils.data import DataLoader
import wandb
wandb.login(key="1a52f6079ddb0d4c0e9f5869d9cc0bdd3f5d9a01")

#from data.from_ac_grammar_vae import print_contact_info
from from_ac_grammar_vae.cfg_equations import CFGEquationDataset
from from_ac_grammar_vae.alphabet import alphabet
from from_ac_grammar_vae.transforms import MathTokenEmbedding, ToTensor, Compose, PadSequencesToSameLength

def main():

    wandb.init(project="Bayesian reasoning in latent space", name="Equation generation")

    params = {
        #"n_samples_data": 100,
        #"n_samples_training": 100000,
        "n_samples" : 2000,
        "batch_size": 256,
       "equations_2000.txt": "equations_2000.txt"
    }
    wandb.config.update(params)

    data = CFGEquationDataset(n_samples=params["n_samples"])
    print("Dataset initialized.")

    data.save("equations_2000.txt")
    print("Dataset saved to file.")

    emb = MathTokenEmbedding(alphabet=alphabet)

    x = data[42]
    x_emb = emb.embed(x)
    print(x)
    print(x_emb)
    print(emb.decode(x_emb))

    wandb.log({
        "example_equation": x,
        "embedded_example": x_emb,
        "decoded_example": emb.decode(x_emb)
    })

    training = CFGEquationDataset(
        n_samples=params["n_samples"],
        transform=Compose([
            MathTokenEmbedding(alphabet),
            ToTensor(dtype=torch.uint8)
        ]))

    training_loader = DataLoader(dataset=training,
                                 batch_size=params["batch_size"],
                                 shuffle=True,
                                 collate_fn=PadSequencesToSameLength())

    print("Training data loaded.")


    for batch_idx, X in enumerate(training_loader):
        print(X.shape)

        eq_dec = emb.decode(X[0], pretty_print=True)
        eq_dec = emb.decode(X[0], pretty_print=True)
        wandb.log({
            "batch_index": batch_idx,
            "batch_shape": X.shape,
            "decoded_example": eq_dec
        })
        print(eq_dec)

    #print_contact_info()

    wandb.finish()


if __name__ == "__main__":
    main()