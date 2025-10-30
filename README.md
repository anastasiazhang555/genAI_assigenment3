# CNN + GAN API (Assignment 3)

This project implements both a CNN image classifier and a GAN generator using PyTorch and FastAPI.

## Project Structure
my_project/
│
├── helper_lib/                 # Core helper library
│   ├── init.py
│   ├── data_loader.py          # Data loading utilities
│   ├── model.py                # CNN + GAN model definitions
│   ├── trainer.py              # Training logic for CNN and GAN
│   ├── generator.py            # GAN image generation (no GUI backend)
│   ├── evaluator.py
│   └── utils.py
│
├── main.py                     # FastAPI application entrypoint
├── requirements.txt            # Python dependencies
├── Dockerfile                  # (Optional) Container build file
│
├── cnn_model.pth               # Pretrained CNN weights
├── gan_generator.pth           # Trained GAN generator
├── gan_discriminator.pth       # Trained GAN discriminator
├── gan_generated.png           # Sample generated image output
└── README.md                   # Project documentation
