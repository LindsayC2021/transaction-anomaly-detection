# transaction-anomaly-detection
Transactional Anomaly Detection Using Isolation Forest Model

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

Precision and recall are much more useful than accuracy because of the extreme class imbalance. A model that labels every transaction as normal would achieve high accuracy but would fail to detect fraud. Recall measures how many true anomalies are detected correctly. Precision measures how many flagged transactions are truly fraudulent. 

Of the total 284, 807 transactions, the model detected 2,849. The number of actual fraud cases in the data is 492. In the initial check, it's apparent that the model did as expected and detected about 1% of the anomalies (considering contamination was set to 0.01). 

## Metrics
Precision: 0.10
Recall:0.587

A Recall of ~59% means the model detects about 6 out of every 10 fraudulent transactions. This is a  very solid result for an unsupervised model at this stage. 

A Precision of 10% means that 1 in every 10 flagged  transactions is actually fraud. That means a lot of false positives. At this stage, this is to be expected since the model trained with no labels or feature engineering. This model is a first stage filter which would reduce the search space significantly and then possibly lead to implementation of a supervised model on the remaining transactions. 

## Limitations and Future Work
* Add a supervised second-stage model on labeled data
* Tune the contamination parameters based on business cost tradeoffs

## How to Run the Project
python src/evaluate.py