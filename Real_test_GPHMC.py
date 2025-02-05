import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import json
from utils import convert_expression
from calculate_likelihood import calculate_log_likelihood_from_gaussian
from LSTM_VAE_Model import LSTMVAE
import argparse
# visualize the GP model fit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from datetime import datetime
import wandb
from matplotlib.colors import LinearSegmentedColormap
import imageio
import os



"here we initialize the GP with a dense grid of points and then we sample from the GP using HMC"

# Initialize WandB
def init_wandb(args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    datapoints_name = os.path.splitext(os.path.basename(args.datapoints))[0]
    wandb.init(
        project=f"Real_test_GPHMC_Model-Fitting_mean=actual_withlog_RBF-{datapoints_name}-{args.initial_position}_test",  # Set project name
        name=f"RUN-{current_time}-arg.datapoints",  # Set run name
        config={
            "model": args.model,
            "vocab_path": args.vocab_path,
            "datapoints": args.datapoints,
            "max_length": args.max_length,
            "grid_size": args.grid_size,
            "grid_size_show": args.grid_size_show,
            "dimension": args.dimension,
            "lengthscale": args.lengthscale,
            "outputscale": args.outputscale,
            "std_likelihood": args.std_likelihood,
            "noise": args.noise,
            "boundsx": args.boundsx,
            "boundsy": args.boundsy,
            "interations in exploration phase": args.iterations_sampling,
            "steps in sampling phase": args.steps,
            "step_size": args.step_size,
            "ground_truth": args.datapoints,
            "start_point": args.initial_position,

        }
    )
    wandb.config.update(args)  # Update hyperparameters in WandB

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, outputscale, noise):
        super(GPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module.constant.data.fill_(0)
        self.mean_module.constant = train_y.mean()  # set the mean to the average of the training data
        #print the mean value
        print(f"Mean value: {self.mean_module.constant}")
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # try different kernel
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        self.covar_module.outputscale = torch.tensor(outputscale)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class EnergyFunction(torch.nn.Module):
    def __init__(self, model, likelihood):
        super(EnergyFunction, self).__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, x):
        self.model.eval()
        self.likelihood.eval()

        with torch.enable_grad():
            predictive_distribution = self.likelihood(self.model(x))
            mean = predictive_distribution.mean
            std = predictive_distribution.stddev

            # energy function (from the paper)
            energy = mean - std

        return energy

class EnergyFunction_sample(torch.nn.Module):
    def __init__(self, model, likelihood):
        super(EnergyFunction_sample, self).__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, x):
        self.model.eval()
        self.likelihood.eval()

        with torch.enable_grad():
            predictive_distribution = self.likelihood(self.model(x))
            mean = predictive_distribution.mean
            std = predictive_distribution.stddev

            # energy function (from the paper)
            energy = mean

        return energy


def hmc_sample(initial_position, energy_function, steps, step_size, num_samples, bounds):
    samples = []

    position = initial_position.clone().detach().requires_grad_(True)

    # print("initial_position", initial_position)

    for _ in range(num_samples):
        # initialize the momentum
        momentum = torch.randn_like(position)
        current_position = position.clone()

        current_position.requires_grad_(True)

        for _ in range(steps):
            energy = energy_function(position)
            # print("energy", energy)
            # print("position", position)

            gradient = torch.autograd.grad(energy, position, create_graph=True, allow_unused=True)[0]

            if gradient is None:
                gradient = torch.zeros_like(position)

            # print("gradient", gradient)
            # print("momentum", momentum)

            # update momentum
            # print("momentum", momentum)
            # print("gradient", gradient)
            momentum = momentum - (step_size * gradient / 2)
            # print("new momentum", momentum)

            # update position
            position = position + (step_size * momentum)

            lower_bounds = bounds[:,0].unsqueeze(0)
            upper_bounds = bounds[:,1].unsqueeze(0)

            position = torch.max(torch.min(position, upper_bounds), lower_bounds)
            # constrain the position to be within the bounds
            # position = torch.clamp(position, *bounds)

            # calculate the gradient of the energy function using torch.autograd.grad
            # print("position", position)
            energy = energy_function(position)
            gradient = torch.autograd.grad(energy, position, create_graph=True, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(position)
            # print("new gradient", gradient)
            momentum = momentum - (step_size * gradient / 2)
            position.requires_grad_(True)

        # calculate the energy for the current position and the proposed position
        current_energy = energy_function(current_position)
        # print("current_energy", current_energy)
        proposed_energy = energy_function(position)
        acceptance_prob = torch.exp(current_energy - proposed_energy).item()
        # print("acceptance_prob", acceptance_prob)

        # accept or reject the new position
        if torch.rand(1).item() < acceptance_prob:
            position = position.clone()
            samples.append(position.clone().detach())

        samples = torch.stack(samples) if samples else initial_position.unsqueeze(0)


    return samples

def uniform_sampling_and_gp_fitting(args):
    # generate grid points
    '''x = np.linspace(args.boundsx[0], args.boundsx[1], args.grid_size)
    y = np.linspace(args.boundsy[0], args.boundsy[1], args.grid_size)
    grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)'''


    # generate grid points
    linear_space = np.linspace(args.boundsx[0],args.boundsx[1], args.grid_size)

    mesh = np.meshgrid(*([linear_space] * args.dimension), indexing='ij')

    grid_points = np.stack(mesh, axis=-1).reshape(-1, 4)

    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)



    # load VAE model and vocab
    VAEmodel = torch.load(args.model)
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    max_length = args.max_length
    likelihood_list = []
    valid_points = []
    failed_points = []
    expressions = []

    # for each grid point, calculate the likelihood
    for z in grid_points_tensor:
        result = score_function_vae(z, VAEmodel, max_length, vocab, args)
        if result is not None:
            likelihood, expression = result
            likelihood_list.append(likelihood)
            valid_points.append(z.tolist())
            expressions.append(expression)
        else:
            failed_points.append(z.tolist())
            expressions.append(None)

    # convert to tensor
    #print the valid points and corresponding likelihood one to one
    #for i in range(len(valid_points)):
        #print(f"Valid point: {valid_points[i]}, Likelihood: {likelihood_list[i]}")
    valid_points_tensor = torch.tensor(valid_points, dtype=torch.float32) if valid_points else torch.empty((0, 2))
    likelihood_tensor = torch.tensor(likelihood_list, dtype=torch.float32) if likelihood_list else torch.empty((0,))

    # GP model fitting
    if valid_points:
        model, likelihood = train_gp_model(valid_points_tensor, likelihood_tensor, args)

    else:
        model, likelihood = None, None

    # visualize the GP model fit
    visualize_gp_model_ini(model, grid_points, valid_points, failed_points, expressions, likelihood_tensor)

    return model, likelihood

# VAE的Score Function
def score_function_vae(z, model, max_length, vocab, args):

    logits, index = model.decoder(z.unsqueeze(0), max_length, start_token_idx=vocab['<start>'])

    # make sure index is a 2D tensor
    if index.dim() == 1:  # if the index is 1D, unsqueeze to make it 2D
        index = index.unsqueeze(0)

    print(f"Index shape before reconstruct_expression: {index.shape}")

    #  reconstruct_expression
    expressions = model.reconstruct_expression(index, from_indices=True)

    # get the first expression
    expression = expressions[0] if len(expressions) > 0 else None

    if expression:
        print(f"Reconstructed expression: {expression}")
    else:
        print(f"No valid expression reconstructed from index: {index}")

    try:
        # transform the expression to math expression
        math_expression = convert_expression(expression)

        # calculate the likelihood
        # get the datapoints file from args
        points_file =  args.datapoints
        likelihood = calculate_log_likelihood_from_gaussian(points_file, math_expression, std=args.std_likelihood)
        return likelihood, expression
    except Exception as e:
        print(f"Error with latent point {z.tolist()}: {e}")
        return None

def score_func_VAE(z, model, max_length, num_layers):
    # Calculate the score function using a VAE model.
    # first decode the latent position z to get the expression
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)
    print("z", z)

    # Generate expression using the decoder
    logits, index = model.decoder(z, max_length, start_token_idx=vocab['<start>'])
    print("expression_index", index)

    # Transfer the index to expression
    expression = model.reconstruct_expression(index, from_indices=True)
    print(f"Generated expression: {expression}")

    if isinstance(expression, list):
        expression = " ".join(expression)

    try:
        # Transform the expression from str to math expression
        math_expression = convert_expression(expression)
        print(f"Math expression: {math_expression}")

        # Calculate the likelihood of the math expression
        target_func = math_expression
        points_file = "datapoints_g4x+2.npy"
        likelihood = calculate_log_likelihood_from_gaussian(points_file, target_func, std=args.std_likelihood)
        print(f"Likelihood: {likelihood}")

        '''# Check if the generated expression is meaningful (e.g., valid mathematical expression)
        if math_expression is None or not math_expression:
            print(f"Skipping invalid expression: {expression}")
            return None  # Skip this iteration if the expression is not meaningful'''

    except Exception as e:
        print(f"Error converting expression {expression}: {e}")
        return None  # Skip this iteration if there's an error in expression conversion


    return likelihood, math_expression

def visualize_gp_model_ini(model, grid_points, valid_points, failed_points, expressions, likelihoods):
    # Create grid for visualization
    x = np.linspace(-5, 5, args.grid_size_show)
    y = np.linspace(-5, 5, args.grid_size_show)
    X, Y = np.meshgrid(x, y)
    visualization_points = np.vstack([X.flatten(), Y.flatten()]).T  # Shape (10000, 2)

    visualization_tensor = torch.tensor(visualization_points, dtype=torch.float32)
    #mean use the mean value of the model
    mean = None  # Use the mean value of the model
    #print(f"Mean value of the model3: {mean}")

    mean = None  # Use the mean value of the model

    if model:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model(visualization_tensor)
            mean = predictions.mean.numpy().reshape(X.shape)  # Ensure it matches grid shape


    # Unique expressions for styling
    unique_expressions = list(set(filter(None, expressions)))  # Exclude None
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_expressions)))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', 'h', '+']
    expression_to_style = {exp: (colors[i], markers[i % len(markers)]) for i, exp in enumerate(unique_expressions)}

    # Create a 2D visualization
    plt.figure(figsize=(20, 16))

    # If GP model exists, draw contour
    if mean is not None:
        contour = plt.contourf(X, Y, mean, levels=100, cmap="coolwarm", alpha=0.7)
        cbar = plt.colorbar(contour, pad=0.1)  # Adjust color bar position
        cbar.set_label("Mean Prediction", fontsize=14)

    # Plot grid points with expressions
    for point, expression in zip(grid_points, expressions):
        if expression is None:  # Failed points
            plt.scatter(point[0], point[1], color="lightgray", edgecolors="k", alpha=0.5)
        else:  # Successful points
            color, marker = expression_to_style[expression]
            plt.scatter(point[0], point[1], color=color, marker=marker, edgecolors="k", alpha=0.8)

    # Add legend for expressions
    handles = [
        plt.Line2D([0], [0], marker=marker, color=color, linestyle="", markersize=10)
        for exp, (color, marker) in expression_to_style.items()
    ]
    plt.legend(
        handles,
        unique_expressions,
        loc="upper left",
        bbox_to_anchor=(1.25, 1),
        fontsize=12,
        title="Expressions",
        title_fontsize=14
    )

    plt.title("GP Model Fit to Uniformly Sampled Points", fontsize=16)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the 2D contour plot to WandB with the lengthscale as the title
    wandb.log({f"2D Contour with lengthscale = {args.lengthscale}": wandb.Image(plt)})


    # Create a 3D plot for GP mean
    if mean is not None:
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mean, cmap="viridis", alpha=0.8)
        ax.set_zlim(mean.min(), mean.max())
        print(f"Mean min: {mean.min()}, Mean max: {mean.max()}")

        # Add labels and title
        ax.set_title("3D Surface of GP Mean Prediction", fontsize=16)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_zlabel("Mean Prediction", fontsize=14)

        # Save to WandB
        wandb.log({f"3D Surface with lengthscale = {args.lengthscale}": wandb.Image(plt)})

    plt.show()

def train_gp_model(train_x, train_y, args):
    #print train x and train y
    print(f"Train x: {train_x}, Train y: {train_y}")
    train_y = train_y.view(-1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(args.noise)

    model = GPModel(train_x, train_y, likelihood, args.lengthscale, args.outputscale, args.noise)
    model.eval()
    #print the mean value of the model
    print(f"Mean value of the model2: {model.mean_module.constant}")
    likelihood.eval()

    return model, likelihood

# Visualize GP model for each lengthscale
"""def visualize_lengthscales(args):
    grid_lengthscales = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # Define lengthscale values to test

    for lengthscale in grid_lengthscales:
        print(f"Visualizing GP with lengthscale: {lengthscale}")
        args.lengthscale = lengthscale  # Update lengthscale in arguments

        # Fit the GP model for the current lengthscale
        model, likelihood = uniform_sampling_and_gp_fitting(args)

        # Visualize the GP model and log the results to WandB
        if model:
            plt.figure(figsize=(10, 8))
            visualize_gp_model_ini(
                model=model,
                grid_points=[],
                valid_points=[],
                failed_points=[],
                expressions=[],
                likelihoods=None,
            )
            wandb.log({"lengthscale": lengthscale, "GP Visualization": wandb.Image(plt)})
            plt.close()  # Close the plot to avoid overlapping"""


def visualize_gp_model_exp(model, accumulated_samples_list, math_expressions_list, args):
    step_size = args.step_size
    steps = args.steps
    # Create a grid for visualization using the specified bounds and grid size
    x = np.linspace(args.boundsx[0], args.boundsx[1], args.grid_size_show)
    y = np.linspace(args.boundsy[0], args.boundsy[1], args.grid_size_show)
    X, Y = np.meshgrid(x, y)

    # Convert grid to a tensor format for model predictions
    inputs = np.vstack([X.flatten(), Y.flatten()]).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Model predictions in batches
    batch_size = 50
    mean_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i + batch_size]
            pred = model(batch_inputs)
            mean_list.append(pred.mean.numpy())

    mean = np.concatenate(mean_list)
    Z = mean.reshape(X.shape)

    # Map each mathematical expression to a unique color
    unique_expressions = list(set(math_expressions_list))
    colors = cm.get_cmap('tab10')
    expression_to_color = {exp: colors(i / len(unique_expressions)) for i, exp in enumerate(unique_expressions)}

    fig = plt.figure(figsize=(16, 12))
    # 3D plot of the mean predictions
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Creating a gradient of colors for the lines connecting the points
    line_colors = [cm.gray(1- i / len(accumulated_samples_list)) for i in range(len(accumulated_samples_list))]

    # Scatter and connect points in 3D plot
    last_point = None
    for idx, (sample, expression) in enumerate(zip(accumulated_samples_list, math_expressions_list)):
        point = np.array(sample)
        if len(point) >= 2:
            z_bottom = Z.min()
            ax_3d.scatter(point[0], point[1], z_bottom, color=expression_to_color[expression], s=20, alpha=0.8)
            if last_point is not None:
                ax_3d.plot([last_point[0], point[0]], [last_point[1], point[1]], [z_bottom, z_bottom],
                           color=line_colors[idx], linewidth=1.5)
            last_point = point

    # Legend for unique mathematical expressions
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_to_color[exp], markersize=10)
               for exp in unique_expressions]
    ax_3d.legend(handles, unique_expressions, loc='upper right', bbox_to_anchor=(1.3, 1))

    ax_3d.set_xlabel('z1')
    ax_3d.set_ylabel('z2')
    ax_3d.set_zlabel('Mean Prediction')
    ax_3d.set_title('3D Visualization of GP Model Mean Predictions and sampling points')

    # 2D contour plot of the mean predictions
    ax_2d = fig.add_subplot(122)
    contour = ax_2d.contourf(X, Y, Z, levels=100, cmap="coolwarm", alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax_2d, pad=0.1)
    cbar.set_label("Mean Prediction", fontsize=14)

    # Scatter and connect points in 2D plot
    last_point = None
    for idx, (sample, expression) in enumerate(zip(accumulated_samples_list, math_expressions_list)):
        point = np.array(sample)
        if len(point) >= 2:
            ax_2d.scatter(point[0], point[1], color=expression_to_color[expression], s=20, edgecolors="k", alpha=0.8)
            if last_point is not None:
                ax_2d.plot([last_point[0], point[0]], [last_point[1], point[1]], color=line_colors[idx], linewidth=1.5)
            last_point = point

    ax_2d.legend(handles, unique_expressions, loc='upper right')
    ax_2d.set_xlabel('z1')
    ax_2d.set_ylabel('z2')
    ax_2d.set_title('2D Contour of GP Model Mean Predictions and sampling trajectory')

    plt.tight_layout()
    # Save the visualization locally and log it to wandb
    file_name = f"gp_visualization_step_{step_size}.png"
    plt.savefig(file_name)
    wandb.log({f"gp_visualization_stepsize_{step_size}_step{steps}": wandb.Image(file_name)})
    plt.close(fig)

def visualize_gp_model_exp_video(model, accumulated_samples_list, math_expressions_list, args):
    # Parameter settings
    step_size = args.step_size
    steps = args.steps
    x = np.linspace(args.boundsx[0], args.boundsx[1], args.grid_size_show)
    y = np.linspace(args.boundsy[0], args.boundsy[1], args.grid_size_show)
    X, Y = np.meshgrid(x, y)
    inputs = np.vstack([X.flatten(), Y.flatten()]).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    batch_size = 50

    # Initialize plotting and video recording
    frames = []
    fig = plt.figure(figsize=(16, 12))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_2d = fig.add_subplot(122)

    # Prediction and plotting
    mean_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i + batch_size]
            pred = model(batch_inputs)
            mean_list.append(pred.mean.numpy())

    mean = np.concatenate(mean_list)
    Z = mean.reshape(X.shape)

    # Create gradient color mapping for line thickness and color
    num_samples = len(accumulated_samples_list)
    gradient_color = LinearSegmentedColormap.from_list('grad_color', ['lightgray', 'black'])
    line_widths = np.linspace(1, 5, num_samples)  # Line width from 1 to 5

    # Plot initial surface
    ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    contour = ax_2d.contourf(X, Y, Z, levels=100, cmap="coolwarm", alpha=0.7)
    plt.colorbar(contour, ax=ax_2d, pad=0.1).set_label("Mean Prediction", fontsize=14)

    # Drawing samples and lines step by step
    for i, sample in enumerate(accumulated_samples_list):
        point = np.array(sample)
        if i > 0:
            previous_point = np.array(accumulated_samples_list[i - 1])
            # Add 3D line
            ax_3d.plot([previous_point[0], point[0]], [previous_point[1], point[1]], [Z.min(), Z.min()],
                       color=gradient_color(i / num_samples), linewidth=line_widths[i])
            # Add 2D line
            ax_2d.plot([previous_point[0], point[0]], [previous_point[1], point[1]],
                       color=gradient_color(i / num_samples), linewidth=line_widths[i])
        # Add scatter for the current point
        ax_3d.scatter(point[0], point[1], Z.min(), color='red', s=20, alpha=0.8)
        ax_2d.scatter(point[0], point[1], color='red', s=20, edgecolors="k", alpha=0.8)

        # Capture the current state
        plt.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # Convert RGBA to RGB
        frames.append(frame)

    # Save the video
    video_path = f"gp_model_visualization_{step_size}_step{steps}.mp4"
    imageio.mimsave(video_path, frames, fps=10)

    # Clean up and close figure
    plt.close(fig)

    # Optionally log to wandb
    wandb.log({"gp_visualization_video": wandb.Video(video_path, fps=4, format="mp4")})

    return video_path

def main():
    # exploration phase
    dimension = args.dimension
    lengthscale = args.lengthscale
    outputscale = args.outputscale
    std = args.std_likelihood
    steps = args.steps
    iterations_sampling = args.iterations_sampling
    step_size = torch.full((dimension,), args.step_size, dtype=torch.float32)
    num_samples = args.num_samples
    bounds = torch.tensor([args.boundsx, args.boundsy], dtype=torch.float32)
    initial_position = torch.tensor(args.initial_position, dtype=torch.float32)

    # initialize the model and likelihood using uniform_sampling_and_gp_fitting
    model, likelihood = uniform_sampling_and_gp_fitting(args)

    model.eval()
    likelihood.eval()
    math_expressions_list = []

    initial_position_sphase = torch.tensor(args.initial_position, dtype=torch.float32)
    sphase_samples = initial_position_sphase.clone()


    total_iterations = iterations_sampling
    burn_in_phase = 300
    expression_count = {}

    for i in range(iterations_sampling):
        energy_function = EnergyFunction_sample(model, likelihood)
        new_samples = hmc_sample(
            sphase_samples[-1:], energy_function, steps, step_size, num_samples, bounds=bounds
        )
        print(f"New samples: {new_samples}")
        new_samples = new_samples.view(initial_position_sphase.shape)

        if i >= burn_in_phase:  # Burn-in phase
            result = score_func_VAE(new_samples, VAEmodel, max_length, num_layers)
            if result is not None:

                Z_score_new, expression = result
                math_expressions_list.append(expression)
                normalized_expression = str(expression)

                # count the number of times each expression is sampled
                if normalized_expression in expression_count:
                    expression_count[normalized_expression] += 1
                else:
                    expression_count[normalized_expression] = 1

                print("normalized_expression", normalized_expression)




        sphase_samples = torch.cat([sphase_samples, new_samples], 0)

        print(f"Iteration {i}")


    #print all the expressions and their corresponding counts
    print(f"Expression count: {expression_count}")

    top_three_expressions = sorted(expression_count.items(), key=lambda item: item[1], reverse=True)[:3]
    print(f"Top three sampled expressions after burn-in: {top_three_expressions}")


    visualize_gp_model_exp(model, sphase_samples, math_expressions_list, args)

    # save the expressions and their counts to a file
    with open(f"expression_count_{args.datapoints}.txt", "w") as f:
        for expression, count in expression_count.items():
            f.write(f"{expression}: {count}\n")

    # save the top three expressions to a file with the name with datapoints

    with open(f"top_three_expressions_{args.datapoints}.txt", "w") as f:
        for expression, count in top_three_expressions:
            f.write(f"{expression}: {count}\n")

    return model, likelihood


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTMVAE_bin_real_test/2025-Feb-04-16-55-52/4/3/model_final.pth")
    parser.add_argument('--vocab_path', type=str, default='LSTMVAE_bin_real_test/2025-Feb-04-16-55-52/4/3/vocab.json')

    #parser.add_argument("--model", type=str, default="model_final.pth")
    #parser.add_argument('--vocab_path', type=str, default='vocab.json')
    parser.add_argument("--datapoints", type=str, default="Nguyen_5_train.npy")

    parser.add_argument("--max_length", type=int, default=37)
    parser.add_argument("--grid_size", type=int, default=20, help="Number of points per axis for uniform sampling")
    parser.add_argument("--grid_size_show", type=int, default=10, help="Number of points per axis for visualization")
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--lengthscale", type=float, default=3.0)
    parser.add_argument("--outputscale", type=float, default=3.0)
    parser.add_argument("--std_likelihood", type=float, default=1, help="likelihood standard deviation for calculating log likelihood")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--boundsx", type=list, default=[-5, 5])
    parser.add_argument("--boundsy", type=list, default=[-5, 5])
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--iterations of gp training", type=int, default=1000)
    parser.add_argument("--number_of_test_points", type=int, default=50)
    parser.add_argument("--iterations_sampling", type=int, default=1000)
    parser.add_argument("--initial_position", type=float, default=[[0, 0]])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=0.015)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--burn_in_steps", type=int, default=300)


    args = parser.parse_args()
    #model, likelihood = uniform_sampling_and_gp_fitting(args)

    # Initialize WandB
    wandb.login(key="1a52f6079ddb0d4c0e9f5869d9cc0bdd3f5d9a01")
    init_wandb(args)

    # Visualize GP model for different lengthscales
    #visualize_lengthscales(args)

    #first initialize the model and likelihood use uniform_sampling_and_gp_fitting
    #model, likelihood = uniform_sampling_and_gp_fitting(args)

    #then start GPHMC exploration phase

    max_length = args.max_length
    num_layers = args.num_layers
    VAEmodel = torch.load(args.model)

    main()

    # Save final model state to WandB
    wandb.finish()
