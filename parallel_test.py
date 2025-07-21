import torch
import torch.nn as nn
import time
import numpy as np

def conv_cudnn():
    # Check if CUDA and cuDNN are available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return
    if not torch.backends.cudnn.enabled:
        print("cuDNN is not enabled. Exiting...")
        return

    # Enable cuDNN and optimize convolution algorithm
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Set device to CUDA
    device = torch.device("cuda")

    # Define input dimensions for heavy convolution
    batch_size, in_channels, height, width = 64, 128, 128, 128
    out_channels = 128
    kernel_size = 3

    # Generate random input data
    input_data = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
    input_tensor = torch.from_numpy(input_data).to(device)

    # Define convolutional layer
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=1
    ).to(device)

    # Warm-up run
    _ = conv(input_tensor)

    # Perform convolution with timing
    start_time = time.time()
    output = conv(input_tensor)
    torch.cuda.synchronize()  # Ensure GPU computation is complete
    end_time = time.time()

    print(f"cuDNN Convolution (batch={batch_size}, {in_channels}->{out_channels}, {height}x{width}) completed in {end_time - start_time:.4f} seconds")
    print(f"Output shape: {output.shape}")
    print(f"Sample output value: {output[0, 0, 0, 0].item():.4f}")

if __name__ == "__main__":
    conv_cudnn()
