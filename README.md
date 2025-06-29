# Image-upscaling-using-Generative-Adversarial-Network
This repository contains a PyTorch implementation of the Residual in Residual Dense Network (RRDN), a deep learning model based on Generative Adversarial Networks (GANs) designed for high-quality image super-resolution. The model upscales low-resolution images to higher resolutions (e.g., 4x upscaling) while preserving fine details and textures.

Project Overview

The RRDN architecture leverages Residual Dense Blocks (RDBs) and Residual in Residual Dense Blocks (RRDBs) to learn hierarchical features for super-resolution tasks. This implementation is inspired by advanced models like ESRGAN, focusing on generating high-quality, visually appealing upscaled images. The code is modular, reusable, and optimized for performance using PyTorch.

Repository: https://github.com/preetham006/Image-upscaling-using-Generative-Adversarial-Network

Features





RRDN Architecture: Combines residual learning and dense connections for robust feature extraction.



Flexible Upscaling: Supports configurable scaling factors (e.g., 4x upscaling).



PyTorch Implementation: Easy-to-use and customizable for research or practical applications.



Modular Design: Includes reusable components like Residual Dense Blocks (RDB) and RRDB.

Installation

Prerequisites





Python 3.8+



PyTorch 2.0+



CUDA-enabled GPU (optional, for faster training/inference)



Additional dependencies: torchvision, numpy, pillow

Setup





Clone the Repository:

git clone https://github.com/preetham006/Image-upscaling-using-Generative-Adversarial-Network.git
cd Image-upscaling-using-Generative-Adversarial-Network



Create a Virtual Environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install torch torchvision numpy pillow

Usage

Training the Model





Prepare the Dataset:





Place low-resolution (LR) and high-resolution (HR) image pairs in a dataset directory.



Example dataset structure:

dataset/
├── train/
│   ├── LR/
│   ├── HR/
├── test/
│   ├── LR/
│   ├── HR/



Run Training: Modify the training script (e.g., train.py) to point to your dataset and configure hyperparameters. Example command:

python train.py --data-dir dataset/ --scale 4 --epochs 100 --batch-size 16



Model Checkpoint: Trained model weights will be saved in the checkpoints/ directory.

Inference

To upscale images using a pre-trained model:

python inference.py --model checkpoints/rrdn_model.pth --input input_lr_image.png --output output_hr_image.png --scale 4

Example

python inference.py --model checkpoints/rrdn_model.pth --input samples/lr_image.png --output samples/hr_image.png --scale 4

This will take a low-resolution image (lr_image.png), upscale it by 4x, and save the result as hr_image.png.

Repository Structure

Image-upscaling-using-Generative-Adversarial-Network/
├── models/
│   └── rrdn.py           # RRDN model implementation
├── train.py              # Training script
├── inference.py          # Inference script
├── dataset/              # Dataset directory (not included, create your own)
├── checkpoints/          # Saved model weights
├── samples/              # Sample input/output images
├── README.md             # This file
└── requirements.txt      # Dependencies

Requirements

See the requirements.txt file for a complete list of dependencies:

torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0

Install them using:

pip install -r requirements.txt

Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Make your changes and commit (git commit -m "Add feature").



Push to the branch (git push origin feature-branch).



Open a Pull Request.

Please ensure your code follows the PEP 8 style guide and includes appropriate documentation.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments





Inspired by the ESRGAN paper: ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.



Built with PyTorch.

Contact

For questions or suggestions, please open an issue on the GitHub repository or contact the maintainer at [your-email@example.com].
