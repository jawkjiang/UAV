"""Quick device check"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Test tensor on GPU
    x = torch.randn(100, 100).cuda()
    print(f"Test tensor device: {x.device}")
    print("\n✓ GPU is working!")
else:
    print("\n✗ Running on CPU (slow)")
    
# Check what the ModelTrainer would use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nModelTrainer will use: {device}")
