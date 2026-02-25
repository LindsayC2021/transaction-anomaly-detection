# transaction-anomaly-detection
Tiered Transactional Anomaly Detection Using Isolation Forest Model

## Project Overview
This project uses an unsupervised anomaly detection model (Isolation Forest) to detect fraudulent transactions in real-world data.

Anomaly detection is the process of separating normal (nonfraudulent) transactions from abnormal (anomalous) transactions in a large dataset. Fraud detection is difficult due to the high stakes involved:
* A false positive incorrectly flags a legitimate customer as fraudulent
* A false negative allows fraudulent transactions to go undetected

In general it is most important to prevent the false negatives since they are more costly and impacts customers and businesses directly. 
In this dataset, the 'Class' feature labels the fraudulent transactions. In practice, such a label isn't available and must be inferred from data alone. 

## Dataset Description
This dataset comes from publicly available data on Kaggle titled "Credit Card Fraud Detection". It contains credit card transactions made by European cardholders in September 2013. 
* Features V1 - V28 were transformed via PCA due to confidentiality concerns
* The only features that were not transformed with PCA are 'Time' and 'Amount' 
* The feature 'Class' is the response variable 
    * 1 represents fraud and 0 represents "normal". 
This is a very imbalanced dataset considering by and large the transactions are "normal" (nonfraudulent). This fact makes it well-suited for an Isolation Forest anomaly detection model!

## Approach
An unsupervised approach was chosen due to the highly imbalanced nature of the dataset. Isolation Forest is a common first choice for anomaly detection because it assumes anomalies are rare and also different enough from normal observations that they can be isolated quickly. The main assumption of Isolation Forest is that anomalies require fewer random splits to isolate than normal data points.

I excluded labels from training so that the model would learn patterns from the data itself and not be given the labels explicitly.  

**Tiered Risk Optimization:**  
To improve operational efficiency and reduce customer friction, transactions are classified into three risk tiers based on anomaly scores:  
* **High** – Auto-flag for immediate investigation  
* **Medium** – Manual review (borderline cases)  
* **Low** – Automatically approved  

High and Medium risk transactions are **reviewed manually**, so legitimate transactions are **not automatically blocked**, maintaining customer experience and reducing churn.

## Project Structure
```text
src/
├── load_data.py                # Loads the dataset
├── preprocess.py               # Prepares and normalizes the features
├── train_model.py              # Trains the Isolation Forest model
└── evaluate.py                 # Evaluates performance
```

## Model Training
The model was trained using all features except 'Class' and a contamination parameter of 0.01. This represents the assumption that 1% of all transactions are anomalous. The model is learning "normal behavior" patterns and what is abnormal without being influenced by 'Class'. 

## Evaluation Results
Isolation Forest outputs predictions as:
*  -1 is anomaly
*   1 is normal transaction

Precision and recall are much more useful than accuracy because of the extreme class imbalance. A model that labels every transaction as normal would achieve high accuracy but would fail to detect fraud. Recall measures how many true anomalies (fraud) are detected correctly. Precision measures how many flagged transactions are truly fraudulent. 

Of the total 284, 807 transactions, the model detected 2,849. The number of actual fraud cases in the data is 492. In the initial check, it's apparent that the model did as expected and detected about 1% of the anomalies (considering contamination was set to 0.01). 

## Baseline Metrics
Precision: 0.10
Recall: 0.59
Fraud caught (TP): 289
False positivies (FP): 2,560
Total flagged: 2,849
Workload reduction: n/a  

## Tiered Risk Optimization
Precision: 0.077
Recall: 0.66
Fraud caught (TP): 327
False positivies (FP): 3,946
Total flagged: 4,273
Workload reduction: 50% fewer transactions flagged

## Limitations and Future Work
* Add a supervised second-stage model on labeled data
* Tune the contamination parameters based on business cost tradeoffs
* Explore additional anomaly detection algorithms for higher sensitivity

## How to Run the Project
python src/evaluate.py