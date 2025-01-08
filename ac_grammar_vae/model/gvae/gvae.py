import torch

from .encoder import GrammarEncoderNetwork
from .decoder import GrammarDecoderNetwork

from ac_grammar_vae.data.transforms import GrammarParseTreeEmbedding
from ac_grammar_vae.model.gvae.interpreter import TorchEquationInterpreter, ExpressionWithParameters

from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf


class GrammarVariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_grammar_productions, max_of_production_steps, latent_dim, rule_embedding: GrammarParseTreeEmbedding, expressions_with_parameters: bool = False):
        super(GrammarVariationalAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.num_grammar_productions = num_grammar_productions
        self.max_of_production_steps =  max_of_production_steps
        self.rule_embedding = rule_embedding

        # initialize the encoder network
        self.encoder = GrammarEncoderNetwork(
                            num_grammar_productions= num_grammar_productions,
                            max_of_production_steps= max_of_production_steps,
                            latent_dim=latent_dim,
                            conv_num_filters=[2,3,4],
                            conv_filter_sizes=[3,3,3],   #[2,3,4],
                            dense_units=[100]
                        )

        # initialize the decoder network
        self.decoder = GrammarDecoderNetwork(
                            num_grammar_productions=num_grammar_productions,
                            max_of_production_steps=max_of_production_steps,
                            conv_filter_sizes=[3, 3, 3] #new added
                        )

        self.multinomial_nll_loss = torch.nn.CrossEntropyLoss()

        # generate the decoder masks from the grammar embedding generation
        self.generate_decoder_masks(rule_embedding=rule_embedding)

        self.expressions_with_parameters = expressions_with_parameters

    def generate_decoder_masks(self, rule_embedding: GrammarParseTreeEmbedding):

        self._decoder_masks = torch.zeros(len(rule_embedding.non_terminals) + 1 ,len(rule_embedding))
        self._non_terminal_idx = {} #initialize the non-terminal index
        self._grammar_start = rule_embedding.start_symbol #initialize the start symbol

        # generate the masks for the non-terminals
        for i, symbol in enumerate(rule_embedding.non_terminals):

            all_embeddings = torch.as_tensor(rule_embedding.embed_all_productions_with(lhs=symbol))
            all_embeddings = torch.nn.functional.one_hot(all_embeddings, num_classes=len(rule_embedding))

            self._decoder_masks[i + 1] = torch.sum(all_embeddings, dim=0)
            self._non_terminal_idx[symbol] = (i + 1)

        # generate the mask for the noop rule on-operation rule
        self._decoder_masks[0] = torch.zeros(len(rule_embedding)) #initialize the mask
        self._decoder_masks[0][0] = 1 #set the mask to 1

        self._embedding_decoder = rule_embedding.decode

    def get_mask_from_lhs_for_one_hot_embedding(self, embedding):
        """
        Given the one-hot embedding
        :return:
        """
        prod_idx = embedding

        prod_idx = torch.as_tensor(torch.nn.functional.one_hot(prod_idx, num_classes=self.num_grammar_productions), dtype=torch.float)
        mask = prod_idx @ self._decoder_masks.T @ self._decoder_masks

        return torch.as_tensor(mask, dtype=torch.bool)

    def encode(self, X, sample=False):

        # check whether the input is encoded as a list of indices which needs to be one-hot-encoded first
        if X.dtype == torch.int64:
            X = torch.as_tensor(torch.nn.functional.one_hot(X, num_classes=self.num_grammar_productions), dtype=torch.float)

        mean, log_std = self.encoder(X)

        if not sample:
            return mean, log_std
        else:
            return mean + torch.rand(mean.shape) * torch.exp(log_std), mean, log_std

    def decode(self, z, sample):
        logits = self.decoder(z)

        if not sample:
            return logits
        else:
            return torch.multinomial(logits, num_samples=1)

    def find_expression_for(self, X, Y, num_opt_steps=10):

        """
        Find the best expression for the given input-output pairs.
        """

        if not self.expressions_with_parameters:
            interpreter = TorchEquationInterpreter()

        # determine bounds for the latent space
        bounds = torch.as_tensor([[-10] * self.latent_dim, [10] * self.latent_dim])

        def score_latent_code(z, max_retries=25, default_score_value=-torch.inf):

            expr = None
            for _ in range(max_retries):
                try:
                    expr = self.sample_decoded_grammar(z=z)
                except:
                    continue

                if expr is not None:

                    # evaluate the expression
                    if self.expressions_with_parameters:
                        parsed_expr = ExpressionWithParameters(expr=expr)
                        parsed_expr.optimize_parameters(X, Y)
                        Y_approx = parsed_expr(X)
                    else:
                        Y_approx = interpreter.evaluate(expr, x=X)

                    # compute the error
                    rmse = torch.sqrt(torch.mean(torch.square(Y - Y_approx)))

                    if torch.isfinite(rmse):
                        return -(rmse), parsed_expr if self.expressions_with_parameters else expr

            return torch.as_tensor(default_score_value), None

        def get_candidate_from_acquisition_func(acq_func):
            z_new, _ = optimize_acqf(
                acq_func,
                bounds=torch.as_tensor([[0.0] * self.latent_dim, [1.0] * self.latent_dim]),
                q=1,
                num_restarts=10,
                raw_samples=256
            )
            z_new = unnormalize(z_new, bounds=bounds)
            return z_new

        def estimate_ucb(Z, Z_score):
            gp = SingleTaskGP(train_X=normalize(Z, bounds), train_Y=Z_score, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            ucb = UpperConfidenceBound(gp, beta=1.0)
            return ucb

        # create initial samples for the latent space
        num_initial_samples = 64
        Z = torch.rand(num_initial_samples, self.latent_dim) * 20 - 10.0

        best_score = -torch.inf
        best_expr = None

        Z_score = []
        for z in torch.split(Z, split_size_or_sections=1, dim=0):
            score, expr = score_latent_code(z, default_score_value=torch.nan)

            if score > best_score:
                best_expr = expr
                best_score = score

            Z_score.append(score)

        Z = torch.as_tensor(Z, dtype=torch.float64)

        Z_score = torch.unsqueeze(torch.stack(Z_score, dim=0), dim=-1)
        Z_score = torch.as_tensor(Z_score, dtype=torch.float64)

        # replace missing values with the worst possible score
        Z_score = torch.nan_to_num(Z_score, nan=torch.min(torch.nan_to_num(Z_score, nan=torch.inf)).item())

        ucb = estimate_ucb(Z.detach(), Z_score.detach())

        for i in range(num_opt_steps):
            z_new = get_candidate_from_acquisition_func(ucb)

            z_score, expr = score_latent_code(z_new)

            if not torch.isfinite(z_score):
                z_score = torch.nan_to_num(z_score, nan=torch.min(Z_score).item())

            if z_score > best_score:
                best_expr = expr
                best_score = z_score

            with torch.no_grad():
                Z = torch.cat([Z, torch.as_tensor(z_new, dtype=torch.float64)])
                z_score = torch.as_tensor(torch.reshape(z_score,shape=[1,1]), dtype=torch.float64)
                Z_score = torch.cat([Z_score, z_score])

            ucb = estimate_ucb(Z.detach(), Z_score.detach())

        return best_expr

    def sample_decoded_grammar(self, z):
        """
        Sample a decoded grammar from the latent space.
        """

        with torch.no_grad():

            if z is None:
                # sample the latent code
                z = torch.randn([1, self.latent_dim])

            logits = self.decode(z, sample=False)
            print("logits: ", logits)

            start_symbol = self._grammar_start
            stack = [ start_symbol ]
            terminals = [ start_symbol ]
            n_steps = 0

            while len(stack) > 0:

                # check whether there are production steps left
                #print("n_steps: ", n_steps)
                #print("max_of_production_steps: ", self.max_of_production_steps)
                if n_steps >= self.max_of_production_steps :
                    print("max_of_production_steps: ", self.max_of_production_steps)
                    #raise Exception("OutOfLogitsException: too many production are required to generate a valid string from the given set of logits.")
                    print("Warning: Reached maximum number of production steps.")
                    break
                # get the first non-terminal from the stack
                nt = stack.pop()

                # generate the mask
                mask = torch.as_tensor(self._decoder_masks[self._non_terminal_idx[nt]], dtype=torch.bool)

                # sample production from masked softmax
                current_logits = torch.masked_fill(logits[:, n_steps, :], mask=~mask, value=float("-inf"))
                print("current_logits: ", current_logits)
                prod_emb = torch.multinomial(torch.softmax(current_logits, dim=-1), num_samples=1)
                #if prod_emb.item() >= len(self._embedding_decoder):
                    #raise Exception("Invalid production index sampled: {}".format(prod_emb.item()))
                prod = self._embedding_decoder(prod_emb.item())

                # find the first occurence
                idx = terminals.index(nt)
                terminals.pop(idx)

                # apply the production
                [terminals.insert(idx, s) for s in reversed(prod.rhs())]
                stack.extend(reversed(list(filter(lambda s: not isinstance(s, str), prod.rhs()))))
                #print the chosen production
                print("chosen production: ", prod)

                n_steps += 1

                # Combine the terminals into a single string (expression)
                expression = ''.join(map(str, terminals))

            return expression

    def negative_elbo(self, X):
        """
        Compute the negative ELBO for the given input batch X.
        """

        batch_size = X.shape[0]

        # autoencode the grammar rules
        z, mean, log_std = self.encode(X, sample=True)
        X_multinom_logits = self.decode(z, sample=False)

        # compute a mask for the multinomial selection (from the inputs X)
        mask = self.get_mask_from_lhs_for_one_hot_embedding(embedding=X)

        # apply the mask to the logits
        X_multinom_logits = torch.masked_fill(X_multinom_logits, mask=~mask, value=float("-inf"))

        # cross-entropy of one-hot vector corresponds to nll
        rec_nll = self.multinomial_nll_loss(torch.reshape(X_multinom_logits, (-1, self.num_grammar_productions)), X.reshape(-1))

        # For the KL computation compare, e.g.
        # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
        kl_divergence = 0.5 * torch.sum(
            -2 * torch.sum(log_std, dim=-1) - self.latent_dim + torch.sum(mean ** 2, dim=-1) + torch.sum(
                torch.exp(log_std) ** 2, dim=-1))

        return (rec_nll + 0.1 * kl_divergence) / batch_size

    def forward(self, X):
        # Encoding step
        _, mean, log_var = self.encode(X, sample=True)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # Assuming start_symbol has its mask at index 0 in _decoder_masks
        current_mask = self._decoder_masks[self._non_terminal_idx[self.rule_embedding.start_symbol]]

        # Decode the latent variable z with the mask
        logits = self.decoder(z, mask=current_mask)  # Pass the mask to the decoder

        return logits

