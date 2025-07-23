import torch
import torch.nn as nn
from copy import deepcopy

# Assume these classes are defined elsewhere
# class MoELLM(nn.Module): ...
# class Router(nn.Module): ...

class SVGD_MoE_Router:
    def __init__(self, full_model: MoELLM, num_particles: int, lr: float):
        self.full_model = full_model # The single, large LLM instance
        self.M = num_particles
        self.lr = lr

        # --- COLLABORATION POINT: Router Initialization ---
        # 1. We only duplicate the router, not the whole LLM.
        # We start by creating N deep copies of the original router.
        # It's better to add small noise to each particle later.
        self.router_particles = [deepcopy(full_model.router) for _ in range(self.M)]

    def _get_router_params(self, router_module: nn.Module) -> torch.Tensor:
        """Flattens the parameters of a router module."""
        return torch.cat([p.view(-1) for p in router_module.parameters()])

    def _set_router_params(self, router_module: nn.Module, flat_params: torch.Tensor):
        """Sets the router module's parameters from a flat tensor."""
        offset = 0
        for param in router_module.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset + numel].view_as(param))
            offset += numel

    def step(self, loss_fn, train_loader):
        # Freeze all parameters in the main LLM except for the router's placeholder
        for name, param in self.full_model.named_parameters():
            if 'router' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Get a batch of data
        x_batch, y_batch = next(iter(train_loader))

        all_log_post_grads = []
        original_router_state = deepcopy(self.full_model.router.state_dict())

        # Loop through each router particle
        for i in range(self.M):
            # 1. "Lego Swap": Load the particle's weights into the main model's router
            self.full_model.router.load_state_dict(self.router_particles[i].state_dict())
            self.full_model.zero_grad()

            # 2. Calculate the gradient for this specific router
            y_hat = self.full_model(x_batch)
            neg_log_likelihood = loss_fn(y_hat, y_batch)
            
            # Simplified prior for brevity
            flat_router_params = self._get_router_params(self.full_model.router)
            neg_log_prior = 0.5 * torch.sum(flat_router_params**2) / len(train_loader.dataset)

            neg_log_posterior = neg_log_likelihood + neg_log_prior
            neg_log_posterior.backward()

            # 3. Store the flattened gradient
            grad_vec = torch.cat([p.grad.view(-1) for p in self.full_model.router.parameters()])
            all_log_post_grads.append(grad_vec)

        # Restore the original router to avoid side effects
        self.full_model.router.load_state_dict(original_router_state)

        # --- SVGD Update Calculation (as before) ---
        stacked_grads = torch.stack(all_log_post_grads)
        
        # Get flattened parameters from our list of router modules
        flat_particles = torch.stack([self._get_router_params(r) for r in self.router_particles])
        
        # ... (RBF kernel and phi calculation logic remains the same) ...
        # kernel_matrix, h = self._rbf_kernel(flat_particles)
        # kernel_grad = ...
        # phi = ...
        
        # For demonstration, let's use a placeholder for the update
        phi = -stacked_grads # A simple SGD-like update without repulsion

        # 4. Apply the update to our standalone router particles
        with torch.no_grad():
            new_particles_flat = flat_particles + self.lr * phi
            for i in range(self.M):
                self._set_router_params(self.router_particles[i], new_particles_flat[i])