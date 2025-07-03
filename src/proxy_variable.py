import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def calculate_rfm(df, snapshot_date=None):
    """
    Calculate Recency, Frequency, Monetary metrics per CustomerId.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'nunique',
        'Amount': 'sum'
    }).reset_index()
    
    rfm.rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }, inplace=True)
    
    return rfm

def create_high_risk_label(rfm_df, n_clusters=3, random_state=42):
    """
    Perform KMeans clustering on scaled RFM features and assign high-risk labels.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Analyze clusters: Least engaged assumed to be high Recency, low Frequency, low Monetary
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    high_risk_cluster = cluster_summary.sort_values(['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).index[0]
    
    rfm_df['is_high_risk'] = np.where(rfm_df['Cluster'] == high_risk_cluster, 1, 0)
    
    return rfm_df[['CustomerId', 'is_high_risk']]

if __name__ == "__main__":
    df = pd.read_csv('../data/data.csv')
    rfm = calculate_rfm(df)
    risk_labels = create_high_risk_label(rfm)

    # Merge with processed training dataset
    processed_df = pd.read_csv('../data/processed_training_data.csv')
    merged_df = pd.merge(processed_df, risk_labels, on='CustomerId', how='left')

    merged_df.to_csv('../data/final_training_data_with_risk.csv', index=False)
    print("Final dataset with high-risk label saved as final_training_data_with_risk.csv in data directory.")