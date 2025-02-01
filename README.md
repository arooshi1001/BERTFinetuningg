# BERT Fine-Tuning for Text Classification
## 📌 Overview

This project demonstrates fine-tuning **BERT (Bidirectional Encoder Representations from Transformers)** for text classification tasks. It also explores traditional machine learning models like Decision Tree & Logistic Regression for comparison.

## 🚀 Features

Fine-tunes a pretrained BERT model for classification.

Implements Decision Tree and Logistic Regression baselines.

Loads and processes a dataset (`data.csv` from Google Drive).

Tokenization and padding to a fixed sequence length.

Performance evaluation using various metrics.

## 📂 Dataset

The dataset (data.csv) is stored in Google Drive and is loaded using pandas.

🛠 Installation & Dependencies

To run this project, install the following dependencies:
```bash
pip install transformers torch sklearn pandas numpy
```
Additionally, if using Google Colab, mount your Drive:
```bash
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
## ⚙️ Usage

Load the dataset:
```bash
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/ml-datasets/data.csv')
```
Preprocess data:

Tokenization using BERT tokenizer.

Padding/truncation to a sequence length of 17.

Train & Evaluate Models:

Train Decision Tree, Logistic Regression, and BERT.

_Compare performance metrics._

## 📊 Results

Text lengths were analyzed, leading to a padding length of 17.

BERT outperforms traditional ML models in classification accuracy.

### 📜 License

This project is released under the MIT License.

## 👥 Contributors

Arooshi Sharma

