Machine Learning Guiding Questions

What are the specific objectives and success criteria for your machine learning model?          Introduction
How can you select the most relevant features for training?                                     Data Intro & Analysis    
Are there any missing values or outliers that need to be addressed through preprocessing?       Data Intro & Analysis
Which machine learning algorithms are suitable for the problem domain?                          Initial Model Development
What techniques are available to validate and tune the hyperparameters?                         Initial Model Development
How should the data be split into training, validation, and test sets?                          Initial Model Development
Are there any ethical implications or biases associated with the machine learning model?        Model Interpretation
How can you document the machine learning pipeline and model architecture for future reference? Conclusion(s)

# 🛍️ ML4 Online Retail Case Study

Machine learning model for customer segmentation using the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

---

## 📌 Project Overview

Clustering: How do customers cluster based on their purchasing behavior (eg., product quantities, unit prices, and transaction frequency)

TODO (adjust and complete): This project explores unsupervised learning techniques to cluster customers based on purchasing behavior - including product quantities, unit prices, and transaction frequency.


**Dataset Summary:**  
TODO (adjust and complete) This transactional dataset contains all purchases made between 01/12/2010 and 09/12/2011 by customers of a UK-based online retailer specializing in unique all-occasion gifts. Many customers are wholesalers.

---

## 🎯 Objectives & Success Criteria

TODO (adjust and complete):
- Identify meaningful customer clusters using unsupervised learning
- Evaluate clustering quality using metrics like silhouette score and intra-cluster variance
- Ensure reproducibility and interpretability of the pipeline
- Document ethical considerations and potential biases

---

## 📊 Data Introduction & Analysis

### Raw Data

Steps
(- 25,900 transactions
- Key fields: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`)

| Variable Name | Role    | Type        | Description                                                                                                                        |
| ------------- | ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| InvoiceNo     | ID      | Categorical | A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode     | ID      | Categorical | A 5-digit integral number uniquely assigned to each distinct product.                                                              |
| Description   | Feature | Categorical | Product name.                                                                                                                      |
| Quantity      | Feature | Integer     | The quantities of each product (item) per transaction.                                                                             |
| InvoiceDate   | Feature | Date        | The day and time when each transaction was generated.                                                                              |
| UnitPrice     | Feature | Continuous  | Product price per unit.                                                                                                            |
| CustomerID    | Feature | Categorical | A 5-digit integral number uniquely assigned to each customer.                                                                      |
| Country       | Feature | Categorical | The name of the country where each customer resides.                                                                               |


### Preprocessing

#### Steps
- Add date columns.
- Handle missing CustomerID.
- Remove invalid InvoiceNo entries.
- Convert dates to useful features (e.g., recency, frequency)
- Add Subtotal column.
- Flag cancellations.
- Remove invalid quantity records.

### Dataset Variables

| Variable Name     | Role     | Type         | Description                                                                 |
|--------------------|----------|--------------|------------------------------------------------------------------------------|
| InvoiceNo          | ID       | Categorical  | A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode          | ID       | Categorical  | A 5-digit integral number uniquely assigned to each distinct product.        |
| Description        | Feature  | Categorical  | Product name.                                                                |
| Quantity           | Feature  | Integer      | The quantity of each product (item) per transaction.                         |
| InvoiceDate        | Feature  | Date         | The date and time when each transaction was generated.                       |
| UnitPrice          | Feature  | Continuous   | Product price per unit.                                                      |
| CustomerID         | Feature  | Categorical  | A 5-digit integral number uniquely assigned to each customer.                |
| Country            | Feature  | Categorical  | The name of the country where each customer resides.                         |
| Year               | Derived  | Integer      | The year extracted from the `InvoiceDate`.                                   |
| Month              | Derived  | Integer      | The month extracted from the `InvoiceDate`.                                  |
| Subtotal           | Derived  | Float        | The total amount for each item (Quantity × UnitPrice).                       |
| CancellationFlag   | Derived  | Categorical  | Indicates whether a transaction is cancelled or matched with another invoice. |


### Exploratory Analysis

TODO (adjust and complete):
- Distribution of purchases by country
- Correlation between quantity and unit price
- RFM (Recency, Frequency, Monetary) analysis for feature engineering

---

## 🤖 Initial Model Development
- TODO (adjust and complete):

### Feature Selection
- TODO (adjust and complete):
- RFM features
- Aggregated purchase behavior per customer

### Algorithms Explored
- TODO (adjust and complete):
- K-Means
- DBSCAN
- Hierarchical Clustering

### Validation & Tuning
- TODO (adjust and complete):
- Elbow method and silhouette score for K-Means
- Grid search for DBSCAN parameters
- PCA for dimensionality reduction

### Data Splitting
- TODO (adjust and complete):
- Not applicable for unsupervised learning, but train/test split used for pipeline testing

---

## 🔍 Model Interpretation
- TODO (adjust and complete):
- Visualize clusters using 2D projections (PCA, t-SNE)
- Analyze cluster characteristics (e.g., high spenders vs. frequent buyers)
- Discuss ethical implications:
  - Bias in country-based segmentation
  - Fairness in targeting strategies

---

## ✅ Conclusions
- TODO (adjust and complete):
- Summary of best-performing clustering approach
- Key insights from customer segments
- Recommendations for business applications
- Limitations and future work

---

## 👥 Team Members

- Brian Ssekalegga  
- Luis Curiel Rojas  
- Meisam Mofidi  
- Minling Lian

---

## 🎥 Interactive Content (could be merged with the team members section)
- TODO (adjust and complete):
Links to demo videos, dashboards, or notebooks (to be added).
