import numpy as np
import torch 
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_kernels_into_one_pt_file(kernels_directory, output_file):
    all_kernels = []
    max_num = max([int(i.name.split('.')[0]) for i in kernels_directory.glob("*.npy")])
    for num in range(max_num + 1):
        kernel_file = kernels_directory / f"{num}.npy"
        kernel = np.load(kernel_file)
        print(kernel.shape)
        kernel = torch.from_numpy(kernel).float()  # Convert to PyTorch tensor
        all_kernels.append(kernel)
    
    print(f"Collected {len(all_kernels)} kernels from {kernels_directory}")
    
    # Concatenate all kernels into a single tensor
    all_kernels_tensor = torch.stack(all_kernels, dim=0).to(device)

    print(f"All kernels tensor shape: {all_kernels_tensor.shape}")
    
    # Save the concatenated tensor to the output file
    torch.save(all_kernels_tensor, output_file)


kernels_directory = Path("learned_kernels")

output_file = "all_kernels.pt"

collect_kernels_into_one_pt_file(kernels_directory, output_file)

