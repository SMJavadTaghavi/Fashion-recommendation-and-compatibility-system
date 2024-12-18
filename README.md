# **Graph Neural Network-Based Recommendation Models for Fashion Compatibility**

## **Repository Overview**
This repository contains implementations and detailed documentation of **GNN-based recommendation models** designed for the Polyvore and IQON datasets. These models leverage multi-modal embeddings and graph-based architectures to improve the quality of recommendations. The repository includes:

1. **Model Architectures**
2. **Data Preparation Pipeline**
3. **Evaluation Metrics**
4. **Literature Comparison**
5. **Dependencies**
6. **Dataset References**
7. **Related Article**

---

## **1. GNN Recommender Models**

### **Models Included**
- **GNN Recommender (GraphSAGE-Based)**: Combines text (BERT) and image (EfficientNet) embeddings for recommendation ranking.
- **Enhanced GNN with BPR Loss**: Focuses on ranking relevance with edge weighting and regularization.
- **Improved GNN for IQON Dataset**: Binary classification model optimized using AUC-ROC.

---

## **2. Data Preparation**

- **Feature Extraction**: Multi-modal embeddings using:
  - Text: **BERT**
  - Image: **EfficientNet**
- **Graph Construction**: Users and items as nodes, user-item interactions as edges.
- **Edge Weighting**: Prioritizes significant interactions (Enhanced GNN model).
- **Positive/Negative Sampling**: Supports BPR Loss optimization.

---

## **3. Evaluation Metrics**

- **MRR (Mean Reciprocal Rank)**: For ranking evaluation (Polyvore GNN).
- **Precision@N**: Measures top-N recommendation accuracy (Enhanced GNN).
- **AUC-ROC**: For binary classification performance (IQON GNN).

---

## **4. Dependencies**

The following libraries must be installed to run the models and scripts:

```bash
pip install torch==1.8.1
pip install torch-geometric
pip install torchvision
pip install transformers
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
```

- **PyTorch**: Framework for building neural networks.
- **torch-geometric**: Implements Graph Neural Network operations.
- **transformers**: Provides pre-trained BERT embeddings.
- **torchvision**: For EfficientNet image processing.
- **pandas** and **numpy**: For data manipulation.
- **scikit-learn**: Evaluation and utility functions.
- **matplotlib**: Visualization of metrics and embeddings.

---

## **5. Dataset Reference**

### **Polyvore Dataset**
- **Source**: [Polyvore Outfits Dataset](https://github.com/uky-ml/visual-compatibility)
- **Description**: Contains outfit combinations, item images, and metadata (titles/descriptions).

### **IQON3000 Dataset**
- **Source**: [IQON3000 Dataset Link](https://github.com/kuplab/IQON3000)
- **Description**: Includes mix-and-match clothing data, user interactions, and visual-textual features.

---

## **6. Related Article**

This repository is inspired by the following research paper:

- **Title**: [Recommendation of Mix-and-Match Clothing by Modeling Indirect Personal Compatibility](https://arxiv.org/abs/1909.12345)
- **Authors**: [Authors' Names, Affiliations]

The methods and data-handling techniques used in this repository closely follow the methodologies presented in the article.



## **7. Conclusion**
This repository provides comprehensive implementations of GNN-based models for fashion compatibility, leveraging graph-based methods and multi-modal embeddings. For a deeper understanding of the methodology, refer to the related article.

---

# Developed by:
  Ali Gharibpur
  S.M.Javad Taghavi
یDeveloped by
