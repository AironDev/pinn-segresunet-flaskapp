
import torch # type: ignore

def simulate_tumor_growth(shape=(240, 240, 160), num_iterations=100, diffusion_rate=0.1):
    # Initialize tumor tensor on CPU
    tumor = torch.zeros(shape, dtype=torch.float32, device='cpu')
    
    # Seed initial tumor at center
    tumor[shape[0]//2, shape[1]//2, shape[2]//2] = 1.0
    
    for _ in range(num_iterations):
        # Apply diffusion equation (simplified)
        tumor_new = tumor.clone()
        tumor_new[1:-1, 1:-1, 1:-1] += diffusion_rate * (
            tumor[:-2, 1:-1, 1:-1] + tumor[2:, 1:-1, 1:-1] +
            tumor[1:-1, :-2, 1:-1] + tumor[1:-1, 2:, 1:-1] +
            tumor[1:-1, 1:-1, :-2] + tumor[1:-1, 1:-1, 2:] -
            6 * tumor[1:-1, 1:-1, 1:-1]
        )
        
        # Clip values to [0, 1]
        tumor = torch.clamp(tumor_new, 0, 1)
    
    return tumor


