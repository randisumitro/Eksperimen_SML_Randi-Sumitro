#!/usr/bin/env python3
"""
Automated Titanic Dataset Preprocessing Script
Author: Randi Sumitro

This script performs automated preprocessing of Titanic dataset and saves the results.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TitanicPreprocessor:
    """Titanic dataset preprocessor class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.fitted = False
        
    def fit_transform(self, df):
        """Fit preprocessing parameters and transform data"""
        logger.info("Starting preprocessing pipeline...")
        df_processed = df.copy()
        
        # Log initial info
        logger.info(f"Initial dataset shape: {df_processed.shape}")
        logger.info(f"Missing values: {df_processed.isnull().sum().sum()}")
        
        # Drop unnecessary columns
        columns_to_drop = ['Name']
        if 'Name' in df_processed.columns:
            df_processed = df_processed.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        logger.info(f"Numeric columns: {list(numeric_columns)}")
        logger.info(f"Categorical columns: {list(categorical_columns)}")
        
        if len(numeric_columns) > 0:
            df_processed[numeric_columns] = self.imputer_num.fit_transform(df_processed[numeric_columns])
            logger.info("Filled missing values in numeric columns")
        
        if len(categorical_columns) > 0:
            df_processed[categorical_columns] = self.imputer_cat.fit_transform(df_processed[categorical_columns])
            logger.info("Filled missing values in categorical columns")
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = self.label_encoder.fit_transform(df_processed[col])
                logger.info(f"Encoded categorical variable: {col}")
        
        # Scale numerical features
        if len(numeric_columns) > 0:
            df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])
            logger.info("Scaled numerical features")
        
        self.fitted = True
        logger.info("Preprocessing completed successfully!")
        return df_processed
    
    def transform(self, df):
        """Transform new data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
            
        df_processed = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['Name']
        if 'Name' in df_processed.columns:
            df_processed = df_processed.drop(columns=columns_to_drop)
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        if len(numeric_columns) > 0:
            df_processed[numeric_columns] = self.imputer_num.transform(df_processed[numeric_columns])
        
        if len(categorical_columns) > 0:
            df_processed[categorical_columns] = self.imputer_cat.transform(df_processed[categorical_columns])
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = self.label_encoder.transform(df_processed[col])
        
        # Scale numerical features
        if len(numeric_columns) > 0:
            df_processed[numeric_columns] = self.scaler.transform(df_processed[numeric_columns])
        
        return df_processed

def load_data():
    """Load Titanic dataset"""
    logger.info("Loading Titanic dataset...")
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    df = pd.read_csv(url)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def save_processed_data(df_processed, preprocessor, output_dir):
    """Save processed data and preprocessor object"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    output_path = os.path.join(output_dir, 'titanic_processed.csv')
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor saved to: {preprocessor_path}")
    
    # Save data info
    info_path = os.path.join(output_dir, 'data_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Titanic Dataset Processing Information\n")
        f.write(f"=====================================\n\n")
        f.write(f"Original dataset shape: {df_processed.shape}\n")
        f.write(f"Columns: {list(df_processed.columns)}\n")
        f.write(f"Data types:\n{df_processed.dtypes}\n\n")
        f.write(f"Missing values after processing: {df_processed.isnull().sum().sum()}\n")
        f.write(f"Target variable (Survived) distribution:\n{df_processed['Survived'].value_counts().to_dict()}\n")
    logger.info(f"Data info saved to: {info_path}")

def main():
    """Main preprocessing pipeline"""
    try:
        # Load data
        df = load_data()
        
        # Initialize preprocessor
        preprocessor = TitanicPreprocessor()
        
        # Process data
        df_processed = preprocessor.fit_transform(df)
        
        # Save results
        output_dir = 'data_preprocessed'
        save_processed_data(df_processed, preprocessor, output_dir)
        
        logger.info("Automated preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing failed. Check logs for details.")
        exit(1)
