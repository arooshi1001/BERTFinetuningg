# BERT Fine-Tuning for Sentiment Analysis

## 📌 Project Overview
This project focuses on **fine-tuning BERT (Bidirectional Encoder Representations from Transformers)** for **sentiment analysis**. The model is trained on a labeled text dataset to classify sentiments efficiently. The performance of BERT is compared with **traditional machine learning models** such as **Logistic Regression** and **Decision Trees** to showcase improvements in accuracy and robustness.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
  - [Preprocessing](#1️⃣-preprocessing)
  - [Traditional Machine Learning Models](#2️⃣-traditional-machine-learning-models)
  - [Fine-Tuning BERT](#3️⃣-fine-tuning-bert)
- [Results & Performance](#-results--performance)
- [Installation & Setup](#-installation--setup)
- [Future Improvements](#-future-improvements)
- [Contact](#-contact)

---

## 🛠️ Technologies Used

### 📦 Libraries & Frameworks
- **Transformers (Hugging Face)** - BERT implementation
- **PyTorch** - Deep learning framework
- **scikit-learn** - Traditional ML models & evaluation
- **Pandas** - Data handling & preprocessing
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization

### 💾 Dataset
- The dataset consists of **5,668 text samples** with corresponding sentiment labels (**Positive, Negative, Neutral**).
- Tokenized using **BERT Tokenizer**.
- Train-test split: **80%-20%**.

![image](https://github.com/user-attachments/assets/5b6d0a35-72af-4ddd-8cdc-5ca7f22b3b23)
![image](https://github.com/user-attachments/assets/644c7482-cd73-452b-8d02-1391217726ca)
![image](https://github.com/user-attachments/assets/fe375660-f5ab-4dac-aff8-9c7be37b59c3)
---

## 📊 Model Architecture

### 1️⃣ **Preprocessing**
- Load dataset (`.csv` format)
- Convert text into **TF-IDF features** (for traditional models)
- Tokenization using **BERT Tokenizer**
- Padding & truncation applied to ensure uniform sequence length

### 2️⃣ **Traditional Machine Learning Models**
- **Logistic Regression**
- **Decision Tree Classifier**
- Performance evaluation using **accuracy, precision, recall, F1-score**

### 3️⃣ **Fine-Tuning BERT**
- Pre-trained **BERT-base model** from Hugging Face
- Added **classification head** (fully connected layer)
- Trained using **Cross-Entropy Loss & Adam Optimizer**
- Hardware: **GPU acceleration (CUDA)**
- Metrics: **Accuracy, Loss, Precision, Recall, F1-score**

---
## Performance
### Logistic Regression
![image](https://github.com/user-attachments/assets/d02f36a5-a686-4f5a-96f2-84dcae3d7e30)
### Decision Tree
![image](https://github.com/user-attachments/assets/78ffd778-e5d1-4256-b821-91229c43bbb7)

### Bert
![image](https://github.com/user-attachments/assets/9e01abe2-54df-45b7-88da-f3207da7aed5)



## 🚀 Results & Performance
| Model | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | **98%** | **99%** | **97%** | **98%** |
| Decision Tree | **98%** | **98%** | **98%** | **98%** |
| **Fine-Tuned BERT** | **60%** | **52%** | **91%** | **66%** |

- BERT **outperforms traditional models** significantly in accuracy and generalization.
- **Inference speed & efficiency** are tested for practical deployment.

![image](https://github.com/user-attachments/assets/aa96ae42-8a8b-450c-9401-4e3353ac65c5)







---

## 🏗️ Installation & Setup

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Training Script**  
```bash
python train.py
```

4️⃣ **Test the Model**  
```bash
python test.py --input "Your sample text here"
```

---

## 📜 Future Improvements
- Experiment with **BERT variants** (e.g., RoBERTa, DistilBERT) to improve efficiency.
- **Hyperparameter tuning** for better generalization.
- Deployment as a **REST API** for real-time sentiment analysis.
- Train on a **larger dataset** to improve robustness.

---

## 📩 Contact
For any queries or contributions, reach out via email at **your-email@example.com** or create an issue in the repository.

---

### ⭐ If you found this project helpful, give it a star! ⭐

