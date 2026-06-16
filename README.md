# Customer-Retention-Analysis
This project analyzes customer retention metrics to understand customer behavior and identify factors influencing churn. It includes visualizations for retention rates, cohort analysis, customer lifetime value (CLV), repeat purchase behavior, and customer segmentation. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CustomerRetentionAnalysis:
    """
    A comprehensive class for analyzing customer retention metrics
    """
    
    def __init__(self, data):
        """
        Initialize with customer data
        
        Args:
            data (pd.DataFrame): DataFrame with columns: customer_id, transaction_date, amount
        """
        self.data = data
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        self.retention_cohorts = None
        self.churn_analysis = None
        
    def calculate_retention_rate(self, period='M'):
        """
        Calculate customer retention rate over time periods
        
        Args:
            period (str): Period for grouping - 'M' for month, 'W' for week, 'D' for day
            
        Returns:
            pd.DataFrame: Retention rates by period
        """
        # Group by period and count unique customers
        customers_per_period = self.data.groupby(
            pd.Grouper(key='transaction_date', freq=period)
        )['customer_id'].nunique()
        
        # Calculate retention rate (current vs previous period)
        retention_rate = customers_per_period.pct_change() * 100
        
        return pd.DataFrame({
            'Period': customers_per_period.index,
            'Active_Customers': customers_per_period.values,
            'Retention_Rate_%': retention_rate.values
        })
    
    def cohort_analysis(self):
        """
        Perform cohort analysis to track customer groups over time
        
        Returns:
            pd.DataFrame: Cohort retention matrix
        """
        # Create cohort dates (first purchase month)
        df_cohort = self.data.copy()
        df_cohort['cohort_date'] = df_cohort.groupby('customer_id')['transaction_date'].transform('min')
        df_cohort['cohort_month'] = df_cohort['cohort_date'].dt.to_period('M')
        df_cohort['transaction_month'] = df_cohort['transaction_date'].dt.to_period('M')
        
        # Calculate month difference
        df_cohort['months_active'] = (df_cohort['transaction_month'] - df_cohort['cohort_month']).apply(lambda x: x.n)
        
        # Create cohort table
        cohort_data = df_cohort.groupby(['cohort_month', 'months_active'])['customer_id'].nunique()
        cohort_data = cohort_data.unstack(fill_value=0)
        
        # Calculate cohort retention rates
        cohort_size = cohort_data.iloc[:, 0]
        retention_table = cohort_data.divide(cohort_size, axis=0) * 100
        
        self.retention_cohorts = retention_table
        return retention_table
    
    def calculate_customer_lifetime_value(self, discount_rate=0.1):
        """
        Calculate Customer Lifetime Value (CLV) for each customer
        
        Args:
            discount_rate (float): Annual discount rate
            
        Returns:
            pd.DataFrame: CLV for each customer
        """
        # Total spending per customer
        customer_spending = self.data.groupby('customer_id').agg({
            'amount': 'sum',
            'transaction_date': ['min', 'max', 'count']
        }).reset_index()
        
        customer_spending.columns = ['customer_id', 'total_spent', 'first_purchase', 'last_purchase', 'transactions']
        
        # Calculate customer lifetime in years
        customer_spending['customer_lifetime_days'] = (
            customer_spending['last_purchase'] - customer_spending['first_purchase']
        ).dt.days
        
        customer_spending['customer_lifetime_years'] = customer_spending['customer_lifetime_days'] / 365
        
        # Calculate average transaction value
        customer_spending['avg_transaction_value'] = customer_spending['total_spent'] / customer_spending['transactions']
        
        # Calculate CLV (simplified: total spent adjusted for discount rate)
        customer_spending['clv'] = customer_spending['total_spent'] / (1 + discount_rate) ** customer_spending['customer_lifetime_years']
        
        return customer_spending
    
    def calculate_repeat_purchase_rate(self):
        """
        Calculate repeat purchase rate and frequency
        
        Returns:
            pd.DataFrame: Repeat purchase metrics
        """
        purchase_counts = self.data.groupby('customer_id').size()
        
        repeat_purchases = (purchase_counts > 1).sum()
        repeat_rate = (repeat_purchases / len(purchase_counts)) * 100
        
        return {
            'total_customers': len(purchase_counts),
            'repeat_customers': repeat_purchases,
            'repeat_purchase_rate_%': repeat_rate,
            'avg_purchases_per_customer': purchase_counts.mean(),
            'max_purchases_by_customer': purchase_counts.max()
        }
    
    def calculate_churn_rate(self, days_inactive=90):
        """
        Calculate churn rate based on inactivity period
        
        Args:
            days_inactive (int): Days to consider a customer as churned
            
        Returns:
            dict: Churn metrics***
