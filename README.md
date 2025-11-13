Machine Learning Guiding Questions (To delete before submission)

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


This project explores focus on unsupervised learning techniques to cluster customers based on purchasing behavior: Recency, Frequency, and Monetary like product quantities, unit prices, and transaction frequency. It applies the machine learning techniques to the online retail dataset. Our goal is to identify meaningful customer segment and create well-separated stable clusters with balance segmentation. The project tries to identify different segments based on RFM metrics to reflect customer behaviour. The distinct customer groups characterized in the project can help the business better understand its products and customers in terms of their profitability. It also turns complex purchasing data into provide customer insights that are easy to interpret to non-technical user.



**Dataset Summary:**  
This transactional dataset contains all purchases made between 01/12/2010 and 09/12/2011 by customers of a UK-based online retailer specializing in unique all-occasion gifts. Many customers are wholesalers. The raw dataset has about 541909 records with eight fields:InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.                         


## 🎯 Objectives & Success Criteria

Objectives:
- Identify meaningful customer clusters using mostly unsupervised learning
- Create well-separated stable clusters
- Evaluate clustering quality using metrics as inertia through the elbow method and cohesion and separation through the silhouette score
- Ensure reproducibility and interpretability of the pipeline
- Document ethical considerations and potential biases

Success Criteria:
-	Clustering achieves a Silhouette Score ≥ 0.6 with low intra-cluster variance
-	Clusters are clearly interpretable and meaningful to business
-	Reducible repreductivity with all preprocessing, feature scaling
-	Unbias segmentation 

## 📊 Data Introduction & Analysis

### Raw Data

Steps:
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
- Add Year & Month columns.
- Handle missing CustomerID.
- Remove invalid InvoiceNo entries.
- Convert dates to useful features. 
- Add Subtotal column.
- Flag cancellations.
- Remove invalid quantity records.

### Dataset Variables

| Variable Name      | Role     | Type         | Description                                                                 |
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
| Subtotal           | Derived  | Continuous   | The total amount for each item (Quantity × UnitPrice).                       |
| CancellationFlag   | Derived  | Categorical  | Indicates whether a transaction is cancelled or matched with another invoice. |

### Dataset Precessing Summary
After our data cleaning, the records in the dataset go down to: 538,945[528,560 with cancelation] from the original: 541,909 records, each for a particular item contained in a
transaction. We added four new columns: Year, Month, Subtotal and CancellationFlat for advance data analysis, and the dataset goes from the original 8 columns to 12 columns. By including new columns, we improve the dataset without modifying the value of the original fields. Therefore we are confident that our data cleaning preserves the characteristic of the original dataset.

### Exploratory Analysis


We used SQLite to extract revelant information for our analysis:

- Total number of valid of transactions after data cleaning:
  19,745 valid transactions. It suggested that average number of distinct products contained in each transaction was 26 (=528,560/19,745). This seemed to suggest that many of the consumers of the business were organizational customers rather than individual customers. Also the majority of the sales were from United Kingdom with 17,863 transactions and total sales of 8.6 million pounds. And the top sale product was "WORLD WAR 2 GLIDERS ASSTD DESIGNS" with a total sales of 54,903 pounds. 
 

- Distribution of the top 10 sales by country

| Country         | Total Transactions | Formatted Revenue |
|----------------|--------------------|-------------------|
| United Kingdom | 17,863             | £8,693,481.34     |
| Netherlands     | 94                | £285,446.34       |
| EIRE            | 281               | £276,150.18       |
| Germany         | 451               | £227,569.65       |
| France          | 386               | £207,743.45       |
| Australia       | 54                | £138,219.71       |
| Spain           | 89                | £61,530.36        |
| Switzerland     | 52                | £56,905.95        |
| Belgium         | 98                | £41,196.34        |
| Japan           | 19                | £37,416.37        |

- Top 10 selling products overall

| StockCode | Description                             | Total Sold |
|-----------|-----------------------------------------|------------|
| 84077     | WORLD WAR 2 GLIDERS ASSTD DESIGNS       | 54,903     |
| 85099B    | JUMBO BAG RED RETROSPOT                 | 47,823     |
| 85123A    | WHITE HANGING HEART T-LIGHT HOLDER      | 37,021     |
| 22197     | POPCORN HOLDER                          | 36,555     |
| 84879     | ASSORTED COLOUR BIRD ORNAMENT           | 36,418     |
| 21212     | PACK OF 72 RETROSPOT CAKE CASES         | 36,310     |
| 23084     | RABBIT NIGHT LIGHT                      | 30,757     |
| 22492     | MINI PAINT SET VINTAGE                  | 26,561     |
| 22616     | PACK OF 12 LONDON TISSUES               | 26,135     |
| 21977     | PACK OF 60 PINK PAISLEY CAKE CASES      | 24,780     |

- Correlation between quantity and unit price
  #feature_analysis file under experiments folder
  #plot image for support in the support images folder

Our initial correlation model showed a correlation value between Quantity and UnitPrice of: -0.0209, which was a weak correlation. There is virtually no linear relationship between Quantity and UnitPrice. A small negative correlation (–0.0209) suggests that, on average, buying more items is very slightly associated with lower unit prices. However, this was a bit counterintuitive and not meaningful based on normal business sense. When we plotted Quantity and UnitPrice chart, we saw there were a couple of high value transactions with lower number of transactions and majority of transactions were at the smaller values, this may explain why without price segmentation, initial correction model showed a weak correlation between unit price and quantity. From the business point of view, these transactions were valid as they were genuine transaction records; however, they may be outliers from the data analysis point of view, and we may need to treat them separately. 


![Online Retail Dataset ](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/correlation_qty_price.png)

### Feature Selection

- RFM (Recency, Frequency, Monetary) and Aggregated purchase behavior per customer for feature engineering
  
Here we also conducted RFM analysis to group customers based on how they shop: Recency, Frequency and Monetary. This helped to understand customers’ buying habits so that business owners can identify who their best and most loyal customers were.
1) Recency represented: How long it’s been since the customer last made a purchase. Customers who bought something recently get higher scores: 5 = very recent, 1 = very old.
2) Frequency represented: How often the customer makes a purchase. People who buy more often get higher scores: 5 = frequent buyer, 1 = rare.
3) Monetary represent: How much money the customer has spent in total. Customers who spend more get higher scores: 5 = high spender, 1 = low spender.

For each customer, we scored their Recency, Frequency and Monetary from 1 to 5: 5 meant the customer performed very well on that measure, 1 meant the customer performed poorly on that measure. 
The we ranked each customer based on their total RFM score: Recency score + Frequency score + Monetary score. Below tables showed the samples RFM results and how to intepret it.

| CustomerID | Recency | Frequency | Monetary | RFM Score | Interpretation                          |
|------------|---------|-----------|----------|-----------|------------------------------------------|
| 12347      | 2       | 7         | £4,310.00| 5-5-5     | Best customer: recent, frequent, high spender |
| 12348      | 75      | 4         | £1,797.24| 2-4-4     | Good spender, moderately active        |
| 12349      | 19      | 1         | £1,757.55| 4-1-4     | High spender, but infrequent           |
| 12350      | 310     | 1         | £334.40  | 1-1-2     | Dormant and low value                 |
| 12352      | 36      | 7         | £2,385.71| 3-5-5     | Active and valuable                    |


## 🤖 Initial Model Development

### Algorithms Explored
- K-Means 
- DBSCAN

By transforming raw transactional data into structured RFM features and applying unsupervised learning, we can identify distinct customer groups that inform marketing, inventory planning, and strategic outreach.

To achieve this, we first engineered RFM features and implemented 2 separated models (K-Means and DBSCAN) to compare customer behaviours, using both the Elbow Method and Silhouette Score to determine the optimal number of clusters. 

However, one customer 'guest-United Kingdom' exhibited extreme purchasing behavior, distorting the clustering results. Removing this outlier allowed for more balanced segmentation and clearer interpretation of typical customer patterns. 

To address this, a second model was run after excluding this outlier, resulting in more balanced clusters and clearer behavioral distinctions. This adjustment ensures that the segmentation reflects typical customer behavior improving interpretability and business relevance.

Together, these models provide a comprehensive view of customer diversity and purchasing dynamics, enabling more informed business decisions.

(Add images to compare improvements from baseline to second model)

### Validation & Tuning

- Elbow method and silhouette score for K-Means
- Cluster visualization

#### K-means
![K-Means Elbow and Silhouette Plot with Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/kmeans_elbow_with.png)
![DBK-MeansSCAN Elbow and Silhouette Plot without Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/kmeans_elbow_without.png)
![K-Means Clusters with Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/kmeans_cluster_with_outlier.png)
![K-Means Clusters without Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/kmeans_cluster_without_outlier.png)

#### DBSCAN 
![DBSCAN Elbow and Silhouette Plot with Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/dbscan_elbow_and_silhouette_with.png)
![DBSCAN Elbow and Silhouette Plot without Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/dbscan_elbow_and_silhouette_without.png)
![DBSCAN Clusters with Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/dbscan_cluster_with_outlier.png)
![DBSCAN Clusters without Outlier](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/dbscan_cluster_without_outlier.png)

## 🔍 Model Interpretation

- Visualize clusters using 2D projections (PCA)
- Analyze cluster characteristics (e.g., high spenders vs. frequent buyers)

![KMeans PCA](https://github.com/lebronbrian23/clustering-online-retail-sales/blob/main/support_images/kmeans_pca_final.png)


K-Means cluster analysis (without outlier):
| Cluster | Monetary (Mean) | Monetary (Median) | Frequency (Mean) | Frequency (Median) | AvgOrderValue (Mean) | AvgOrderValue (Median) |
|---------|------------------|-------------------|-------------------|---------------------|------------------------|-------------------------|
| 0       | £556.43          | £310.47           | 1.57              | 1.0                 | £344.72                | £230.35                 |
| 1       | £1,871.11        | £919.61           | 4.66              | 3.0                 | £388.49                | £308.18                 |
| 2       | £82,425.61       | £59,557.62        | 69.54             | 56.0                | £1,731.74              | £1,152.23               |

<!-- ![DBSCAN PCA](To add) -->

---

## ✅ Conclusions

- Summary of best-performing clustering approach:

- Key insights from customer segments:

- Recommendations for business applications:

- Limitations and future work: Deployment of models on to the cloud using AWS or GCP. Implement supervised models to identify loyal customers (will need to set some labeling prior model implementation). Create interactive website for further analysis.


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
