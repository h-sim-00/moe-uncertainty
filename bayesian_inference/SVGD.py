import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

# Helper function to flatten model parameters into a single vector
def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

# Helper function to set model parameters from a flat vector
def set_flat_params(model, flat_params):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        # Use .data to avoid tracking this operation in the graph
        param.data.copy_(flat_params[offset:offset + numel].view_as(param))
        offset += numel

class SVGD_PyTorch(Optimizer):
    """
    Implements SVGD for a PyTorch model.
    Note: This inherits from PyTorch's base Optimizer for structure, but the
    step logic is custom and doesn't use the typical optimizer.step().
    We manage parameters manually.
    """
    def __init__(self, model_class, num_particles, lr=1e-3, **model_args):
        # We don't use the `params` argument of the base Optimizer
        # Instead we manage a list of particle tensors.
        # The `defaults` dict is required by the base class.
        defaults = dict(lr=lr)
        super(SVGD_PyTorch, self).__init__([torch.zeros(1)], defaults)

        self.model_class = model_class
        self.model_args = model_args
        self.M = num_particles
        self.lr = lr

        # Initialize M particles (models) and their optimizers
        self.particles_models = [self.model_class(**self.model_args) for _ in range(self.M)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.particles_models]

    def _rbf_kernel(self, X, h=-1.):
        """
        Computes the RBF kernel matrix and its gradient.
        X: A tensor of particles, shape (M, num_params)
        h: bandwidth. If -1, use the median heuristic.
        """
        XY = X @ X.t()
        X2 = (X**2).sum(1).view(-1, 1)
        # Pairwise squared distances
        D2 = X2 + X2.t() - 2 * XY

        if h == -1:
            # Median heuristic
            # Detach to avoid this calculation being part of the graph
            h = torch.sqrt(0.5 * D2.median() / torch.log(torch.tensor(self.M, dtype=torch.float32)))

        K = torch.exp(-D2 / h**2 / 2)
        return K, h

    def step(self, loss_fn, train_loader):
        """
        Performs a single SVGD update step.
        """
        # Get a batch of data
        try:
            x_batch, y_batch = next(self._data_iterator)
        except (AttributeError, StopIteration):
            self._data_iterator = iter(train_loader)
            x_batch, y_batch = next(self._data_iterator)

        all_log_post_grads = []

        # --- COLLABORATION POINT #1: Gradient Calculation Loop ---
        # My understanding is that for each particle, we need the gradient of the
        # log-posterior (log-likelihood + log-prior). Here's how I'd do that
        # for all particles first, before calculating the kernel.
        for i in range(self.M):
            model = self.particles_models[i]
            model.zero_grad()

            # --- Log-Likelihood Part ---
            y_hat = model(x_batch)
            # The loss_fn should return the NEGATIVE log-likelihood.
            # E.g., nn.CrossEntropyLoss() for classification.
            # Calling .backward() on a negative value makes it a gradient ascent step.
            neg_log_likelihood = loss_fn(y_hat, y_batch)

            # --- Log-Prior Part ---
            # Assuming a standard Gaussian prior N(0,1) over the weights.
            flat_params = get_flat_params(model)
            # The log_prob of N(0,1) is proportional to -0.5 * ||w||^2
            # The gradient of the log-prior is just -w.
            # Scaling by num_datapoints is common in Bayesian DL literature.
            num_datapoints = len(train_loader.dataset)
            neg_log_prior = 0.5 * torch.sum(flat_params**2) / num_datapoints

            # --- Total Log-Posterior ---
            # We want to ascend the log-posterior, which is equivalent to
            # descending the negative log-posterior.
            neg_log_posterior = neg_log_likelihood + neg_log_prior
            neg_log_posterior.backward()

            # Store the flattened gradient of the negative log posterior
            grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
            all_log_post_grads.append(grad_vec)

        # Stack all gradients and all particle weights into tensors
        stacked_grads = torch.stack(all_log_post_grads)
        
        flat_particles = torch.stack([get_flat_params(m) for m in self.particles_models])

        # --- COLLABORATION POINT #2: Kernel and SVGD Update ---
        # The TensorFlow code calculated the kernel gradient using GradientTape on the
        # sum of the kernel, which is clever but can be inefficient. A more direct
        # way is to use the analytical gradient of the RBF kernel.
        kernel_matrix, h = self._rbf_kernel(flat_particles)
        
        # grad(K) = - K * (x - y) / h^2
        kernel_grad = -torch.matmul(kernel_matrix, flat_particles)
        kernel_grad += (kernel_matrix.sum(1, keepdim=True) * flat_particles)
        kernel_grad /= (h**2)

        # The SVGD update rule
        # We use -stacked_grads because they are gradients of the *negative* log posterior.
        # phi = (K * grad(log p)) -> (K @ -neg_grad(log_p))
        phi = (torch.matmul(kernel_matrix, -stacked_grads) + kernel_grad) / self.M

        # --- COLLABORATION POINT #3: Applying the Update ---
        # Instead of a separate optimizer for each particle, we can manage the
        # updates directly since SVGD gives us the final update vector `phi`.
        # We add `phi` to our particles to perform gradient ascent.
        with torch.no_grad():
            new_particles = flat_particles + self.lr * phi
            
            # Pack the updated weights back into each model
            for i in range(self.M):
                set_flat_params(self.particles_models[i], new_particles[i])