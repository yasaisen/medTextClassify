# medTextClassify
Medical text binary classification task

## environment setup
```bash
conda create --name classNLP python=3.10
conda activate classNLP

git clone https://github.com/yasaisen/medTextClassify.git
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3, matplotlib

```

## how to use
```bash
python data_download_and_save.py
python train.py
```
