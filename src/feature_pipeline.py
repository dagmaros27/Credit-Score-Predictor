import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engineering import AggregateFeatures, DateFeatures

def create_feature_pipeline():
    numeric_features = ['Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_count', 
                        'Value_sum', 'Value_mean', 'Value_std']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 
                            'ProductCategory', 'ChannelId', 'PricingStrategy']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor




if __name__ == "__main__":
    DATA_PATH = '../data/data.csv'
    df = pd.read_csv(DATA_PATH)  
    df = DateFeatures().fit_transform(df)
    agg_df = AggregateFeatures().fit_transform(df)

    # Merge aggregate features back with unique customer-level data
    unique_customers = df.drop_duplicates(subset='CustomerId')[['CustomerId', 'CurrencyCode', 'CountryCode', 'ProviderId', 
                            'ProductCategory', 'ChannelId', 'PricingStrategy']]
    merged_df = pd.merge(agg_df, unique_customers, on='CustomerId', how='left')

    # Preserve CustomerId separately
    customer_ids = merged_df['CustomerId'].reset_index(drop=True)

    # Apply preprocessing pipeline
    feature_data = merged_df.drop(columns=['CustomerId'])
    pipeline = create_feature_pipeline()
    final_features = pipeline.fit_transform(feature_data)

    # Combine CustomerId with transformed features
    processed_df = pd.DataFrame(final_features.toarray()) if hasattr(final_features, "toarray") else pd.DataFrame(final_features)
    processed_df['CustomerId'] = customer_ids

    # Save processed dataset
    processed_df.to_csv('../data/processed_training_data.csv', index=False)
    print("Processed training data with CustomerId saved as processed_training_data.csv")

