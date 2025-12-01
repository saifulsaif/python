import pandas as pd


class DataLoader:
    """Class for loading and exploring Titanic data"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Load the CSV data"""
        print("\n[STEP 1] LOADING DATA")
        print("-"*80)

        self.df = pd.read_csv(self.filepath)

        print("✓ Data loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")

        return self.df

    def explore_data(self):
        """Explore and display data information"""
        print("\n[STEP 2] EXPLORING DATA")
        print("-"*80)

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nTarget Distribution (Survived):")
        print(self.df['Survived'].value_counts())
        print(f"Survival Rate: {self.df['Survived'].mean()*100:.2f}%")

    def analyze_data(self):
        """Perform data analysis"""
        print("\n[STEP 3] DATA ANALYSIS")
        print("-"*80)

        print("\n1. Survival by Gender:")
        print(self.df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']))

        print("\n2. Survival by Class:")
        print(self.df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']))

        print("\n3. Age by Survival:")
        print(self.df.groupby('Survived')['Age'].describe())

    def get_dataframe(self):
        """Return the loaded dataframe"""
        return self.df
