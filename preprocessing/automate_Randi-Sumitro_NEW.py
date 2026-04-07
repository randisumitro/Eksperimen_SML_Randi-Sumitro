#!/usr/bin/env python3
"""
Automated Titanic Dataset Preprocessing Script
Author: Randi Sumitro
Project: Membangun Sistem Machine Learning - Dicoding

This script automates the entire preprocessing pipeline for Titanic dataset,
including data loading, cleaning, feature engineering, and saving processed data.
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TitanicPreprocessor:
    """
    Complete Titanic dataset preprocessing class
    Handles missing values, encoding, scaling, and feature engineering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.age_imputer = SimpleImputer(strategy='median')
        self.fare_imputer = SimpleImputer(strategy='median')
        self.feature_columns = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
        self.target_column = 'Survived'
        self.processing_date = datetime.now()
        
    def load_data(self, url=None):
        """Load Titanic dataset from URL or local file"""
        try:
            if url is None:
                url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
            
            logger.info(f"Loading dataset from: {url}")
            df = pd.read_csv(url)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        logger.info("=== Exploratory Data Analysis ===")
        
        # Basic info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        logger.info(f"Missing values:\n{missing_values[missing_values > 0]}")
        
        # Target distribution
        if self.target_column in df.columns:
            survival_rate = df[self.target_column].mean() * 100
            logger.info(f"Survival rate: {survival_rate:.2f}%")
        
        return df
    
    def fit(self, X, y=None):
        """Fit the preprocessor on training data"""
        logger.info("Fitting preprocessor...")
        
        # Make a copy to avoid modifying original data
        X_processed = X.copy()
        
        # Encode Sex column
        if 'Sex' in X_processed.columns:
            X_processed['Sex'] = self.label_encoder.fit_transform(X_processed['Sex'])
            logger.info("Encoded Sex column")
        
        # Fit imputers
        if 'Age' in X_processed.columns:
            self.age_imputer.fit(X_processed[['Age']])
            logger.info("Fitted Age imputer")
        
        if 'Fare' in X_processed.columns:
            self.fare_imputer.fit(X_processed[['Fare']])
            logger.info("Fitted Fare imputer")
        
        # Impute missing values for fitting
        if 'Age' in X_processed.columns:
            X_processed['Age'] = self.age_imputer.transform(X_processed[['Age']])
        
        if 'Fare' in X_processed.columns:
            X_processed['Fare'] = self.fare_imputer.transform(X_processed[['Fare']])
        
        # Fit scaler
        if all(col in X_processed.columns for col in self.feature_columns):
            self.scaler.fit(X_processed[self.feature_columns])
            logger.info("Fitted StandardScaler")
        
        return self
    
    def transform(self, X):
        """Transform the data using fitted preprocessor"""
        logger.info("Transforming data...")
        
        # Make a copy
        X_processed = X.copy()
        
        # Encode Sex column
        if 'Sex' in X_processed.columns:
            X_processed['Sex'] = self.label_encoder.transform(X_processed['Sex'])
        
        # Impute missing values
        if 'Age' in X_processed.columns:
            X_processed['Age'] = self.age_imputer.transform(X_processed[['Age']])
        
        if 'Fare' in X_processed.columns:
            X_processed['Fare'] = self.fare_imputer.transform(X_processed[['Fare']])
        
        # Scale features
        if all(col in X_processed.columns for col in self.feature_columns):
            X_processed[self.feature_columns] = self.scaler.transform(X_processed[self.feature_columns])
            logger.info("Applied StandardScaler")
        
        return X_processed
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data"""
        return self.fit(X, y).transform(X)
    
    def save_processed_data(self, X_processed, y=None, output_dir='data_preprocessed'):
        """Save processed data and preprocessor object"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
            # Combine features and target if available
            if y is not None:
                processed_df = X_processed.copy()
                processed_df[self.target_column] = y
            else:
                processed_df = X_processed
            
            # Save processed dataset
            output_file = os.path.join(output_dir, 'titanic_processed.csv')
            processed_df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to: {output_file}")
            
            # Save preprocessor object
            preprocessor_file = os.path.join(output_dir, 'preprocessor.pkl')
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Saved preprocessor to: {preprocessor_file}")
            
            # Save data information
            info_file = os.path.join(output_dir, 'data_info.txt')
            with open(info_file, 'w') as f:
                f.write(f"Titanic Dataset Preprocessing Results\n")
                f.write(f"Author: Randi Sumitro\n")
                f.write(f"Processing Date: {self.processing_date}\n\n")
                f.write(f"Original shape: {X_processed.shape}\n")
                f.write(f"Features: {self.feature_columns}\n")
                f.write(f"Target: {self.target_column}\n")
                f.write(f"Missing values handled: Age, Fare\n")
                f.write(f"Categorical encoded: Sex\n")
                f.write(f"Features scaled: StandardScaler\n")
                f.write(f"\nPreprocessing completed successfully!\n")
            logger.info(f"Saved data info to: {info_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False

def main():
    """Main function to run the automated preprocessing pipeline"""
    logger.info("=" * 60)
    logger.info("TITANIC DATASET AUTOMATED PREPROCESSING")
    logger.info("Author: Randi Sumitro")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = TitanicPreprocessor()
        
        # Load data
        df = preprocessor.load_data()
        
        # Explore data
        preprocessor.explore_data(df)
        
        # Split features and target
        X = df[preprocessor.feature_columns]
        y = df[preprocessor.target_column]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
        logger.info(f"Processed features shape: {X_processed.shape}")
        
        # Save processed data
        success = preprocessor.save_processed_data(X_processed, y)
        
        if success:
            logger.info("=" * 60)
            logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            # Print summary
            logger.info(f"Total passengers processed: {len(df)}")
            logger.info(f"Survival rate: {df['Survived'].mean()*100:.1f}%")
            logger.info(f"Male passengers: {(df['Sex']=='male').sum()} ({(df['Sex']=='male').mean()*100:.1f}%)")
            logger.info(f"Female passengers: {(df['Sex']=='female').sum()} ({(df['Sex']=='female').mean()*100:.1f}%)")
            
            # Verify saved files
            output_dir = 'data_preprocessed'
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                logger.info(f"Files saved in {output_dir}:")
                for file in files:
                    logger.info(f"  - {file}")
            
            return True
        else:
            logger.error("Preprocessing failed during save operation")
            return False
            
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("AUTOMATED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the 'data_preprocessed' directory for results:")
        print("- titanic_processed.csv")
        print("- preprocessor.pkl") 
        print("- data_info.txt")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PREPROCESSING FAILED!")
        print("Check the log file 'preprocessing.log' for details")
        print("="*60)
