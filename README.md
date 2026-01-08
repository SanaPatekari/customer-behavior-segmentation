# Customer Behavior Segmentation

This repository contains a customer behavior segmentation project built using transaction-level credit card data.  
The project focuses on understanding customer spending patterns using feature engineering and unsupervised learning.

The primary goal is exploratory analysis and segmentation.

---

## Project Objective

- Transform transaction-level data into customer-level behavioral features  
- Group customers with similar behavior patterns  
- Interpret and explore segments in an interactive way  

---

## Dataset

This project uses a publicly available transaction-level credit card dataset from Kaggle.

- **Dataset:** Credit Card Transactions Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/kartik2112/fraud-detection  

The dataset contains transaction records with columns such as:
- Transaction date and time  
- Transaction amount  
- Merchant and category  
- Cardholder and merchant location  

Personally identifiable fields are not used for analysis or visualization.  
The fraud label included in the dataset is not used in this project.

---

## Approach

- Aggregate transactions by customer (`cc_num`)  
- Engineer behavior-based features  
- Scale features for fair comparison  
- Apply clustering to identify customer segments  
- Interpret segments using summary statistics  
- Visualize results using a Streamlit application  

---

## Feature Engineering

Customer-level features include:
- Transaction count  
- Total and average spend  
- Spending volatility  
- Unique merchants and categories  
- Weekend and night transaction share  
- Average and maximum transaction distance  
- Activity duration and transactions per day  

These features describe behavior.

---

## Model Choice

### Why K-Means

K-Means clustering was selected because:
- It is simple and widely used for customer segmentation  
- It works well with numeric, scaled features  
- Results are easy to interpret  
- It scales efficiently to large datasets  

K-Means is used as a baseline method for exploratory segmentation.

---

## Choosing the Number of Clusters

The number of clusters was selected using the Silhouette Score.

- Multiple values of `k` were evaluated  
- The highest silhouette score was observed at `k = 2`  
- Silhouette score at `k = 2`: **0.701**  

This indicates strong separation between the resulting customer segments.  
The choice of two clusters was kept to prioritize interpretability.

---

## Streamlit Application

The Streamlit app allows interactive exploration of results.

### Features

- Portfolio overview of customer segments  
- Segment-level behavior summaries  
- Individual customer behavior profiles  
- Monthly spend trends and category breakdowns  

The UI focuses on clarity and explainability.

---


## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## How to Run

```bash
pip install -r requirements.txt
python src/build_features.py
python src/train_segments.py
streamlit run app.py

