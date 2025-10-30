import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch

def generate_samples(generator, device='cpu', num_samples=10):
    generator.eval()
    z = torch.randn(num_samples, 100, device=device)
    with torch.no_grad():
        samples = generator(z).cpu()

    samples = (samples + 1) / 2  
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')

    output_path = "gan_generated.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✅ Saved generated images to {output_path}")
    return output_path