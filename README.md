# Diversity-Driven Ensemble Learning for Time Series Classification

This repository provides PyTorch implementations of a diversity-driven ensemble learning framework for time series classification (TSC). The approach builds on the **LITE** architecture to create ensembles of neural networks whose feature representations are explicitly decorrelated to improve generalization and classification performance.

The framework has been evaluated on the **UCR Time Series Archive** datasets, demonstrating state-of-the-art accuracy with fewer ensemble members compared to traditional methods.

## Features

- **LITE Model**: Lightweight Inception-based time series classifier.
- **Decorrelated Training**: Penalizes redundancy in learned features across ensemble members.
- **Flexible Ensemble Sizes**: Includes scripts to train ensembles of different sizes (2 to 5 models).
- **Reproducible Experiments**: Configurable via command-line arguments.

## Repository Structure

\`\`\`
.
├── base.py              # Base model training script
├── cotrain.py           # Decorrelated training with 1 reference model
├── cotrain_2.py         # Decorrelated training with 2 reference models
├── cotrain_3.py         # Decorrelated training with 3 reference models
├── cotrain_4.py         # Decorrelated training with 4 reference models
├── cotrain_5.py         # Decorrelated training with 4 reference models (variant)
├── lite.py              # LITE model definition
├── utils.py             # Data loading, preprocessing, and utilities
\`\`\`

## Installation

1. **Clone this repository**

   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   \`\`\`

2. **Install dependencies**

   These scripts require:
   - Python 3.8+
   - PyTorch
   - NumPy
   - Pandas
   - scikit-learn
   - Matplotlib

   You can install them via pip:

   \`\`\`bash
   pip install torch numpy pandas scikit-learn matplotlib
   \`\`\`

## Usage

All training scripts support command-line arguments:

- \`--classifier\`: Currently supports only \`LITE\`
- \`--datasets\`: Name of the dataset (or multiple datasets) from the UCR archive
- \`--runs\`: Number of runs to perform
- \`--output-directory\`: Where to store results

Example to train a single LITE model (baseline):

\`\`\`bash
python base.py --datasets ECGFiveDays --runs 1 --output-directory results/base/
\`\`\`

Example to train a decorrelated ensemble with one reference model:

\`\`\`bash
python cotrain.py --datasets ECGFiveDays --runs 1 --output-directory results/ensemble_1/
\`\`\`

Example to train a decorrelated ensemble with two reference models:

\`\`\`bash
python cotrain_2.py --datasets ECGFiveDays --runs 1 --output-directory results/ensemble_2/
\`\`\`

...and so on up to \`cotrain_5.py\`.

**Note**: You must set up your UCR Archive dataset folder paths in \`utils.py\` (see \`load_data()\`).

## Results

This framework was evaluated on 128 UCR datasets and achieved:

- Improved accuracy compared to standard LITE ensembles
- Higher feature diversity as measured by feature orthogonality metrics

See the paper for detailed quantitative and qualitative analysis:

**[Enhancing Time Series Classification with Diversity-Driven Neural Network Ensembles](./IJCNN_2025_Ensemble_Learning_for_TSC.pdf)**

## Citation

If you use this code, please cite:

\`\`\`
@inproceedings{abdullayev2025diversity,
  title={Enhancing Time Series Classification with Diversity-Driven Neural Network Ensembles},
  author={Abdullayev, Javidan and Devanne, Maxime and Meyer, Cyril and Ismail-Fawaz, Ali and Weber, Jonathan and Forestier, Germain},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2025}
}
\`\`\`

## License

This repository is released under the MIT License.

## Acknowledgments

This work was conducted at Université de Haute Alsace and supported by the IRIMAS laboratory.