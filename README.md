# medTextClassify
Medical text binary classification task

`medTextClassify` is a lightweight framework for training a binary classifier on medical text. It provides utilities to download PubMed abstracts, build a dataset and train a BioBERT-based model.

## Features
- Script to download PubMed articles based on PMIDs
- PyTorch dataset wrapper (`MedicalTextDataset`)
- Simple `BioBERTClassifier` for binary classification
- Training loop with metrics tracking and visualizations

## Installation
1. Create a Python environment (tested with Python 3.10):
   ```bash
   conda create --name classNLP python=3.10
   conda activate classNLP
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yasaisen/medTextClassify.git
   cd medTextClassify
   ```
3. Install the dependencies:
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.51.3 matplotlib seaborn scikit-learn pandas tqdm metapub
   ```
## Usage
1. **Prepare the dataset** (requires network access to PubMed):
   ```bash
   python data_download_and_save.py
   ```
   This creates `dataset/data_*.json` with article titles, abstracts and labels.

2. **Train the model**:
   ```bash
   python train.py
   ```
   Training outputs (model weights, metrics, plots) are stored under `outputs/<timestamp>`.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.






