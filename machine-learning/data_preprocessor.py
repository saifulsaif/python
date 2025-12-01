import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Class for preprocessing and feature engineering"""

    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def preprocess_data(self):
        """Handle missing values and encode categorical variables"""
        print("\n[STEP 4] PREPROCESSING DATA")
        print("-"*80)

        # Handle missing values
        if self.df.isnull().sum().sum() > 0:
            self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
            print("✓ Filled missing Age values")

        # Encode Sex: male=0, female=1
        self.df['Sex_numeric'] = self.df['Sex'].map({'male': 0, 'female': 1})
        print("✓ Encoded Sex column")

        # One-hot encode Embarked
        self.df = pd.get_dummies(self.df, columns=['Embarked'], prefix='Embarked')
        print("✓ One-hot encoded Embarked column")

        return self.df

    def feature_engineering(self):
        """Create new features"""
        print("\n[STEP 5] FEATURE ENGINEERING")
        print("-"*80)

        # Create new features
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        self.df['FarePerPerson'] = self.df['Fare'] / self.df['FamilySize']

        print("✓ Created FamilySize feature")
        print("✓ Created IsAlone feature")
        print("✓ Created FarePerPerson feature")

        print("\nSample of new features:")
        print(self.df[['SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FarePerPerson']].head())

        return self.df

    def select_features(self):
        """Select and prepare features for modeling"""
        print("\n[STEP 6] SELECTING FEATURES")
        print("-"*80)

        # Define feature columns
        self.feature_columns = [
            'Pclass',
            'Sex_numeric',
            'Age',
            'SibSp',
            'Parch',
            'Fare',
            'Embarked_C',
            'Embarked_Q',
            'Embarked_S',
            'FamilySize',
            'IsAlone',
            'FarePerPerson'
        ]

        # Separate features and target
        X = self.df[self.feature_columns]
        y = self.df['Survived']

        print(f"Features selected: {len(self.feature_columns)}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        print("\nFeatures:")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i}. {col}")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n[STEP 7] SPLITTING DATA")
        print("-"*80)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Training set: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"Testing set: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")

        print("\nTraining set survival rate:", self.y_train.mean())
        print("Testing set survival rate:", self.y_test.mean())

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """Scale features using StandardScaler"""
        print("\n[STEP 8] SCALING FEATURES")
        print("-"*80)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✓ Features scaled using StandardScaler")
        print("  Training data: fitted and transformed")
        print("  Testing data: transformed only")

        return self.X_train_scaled, self.X_test_scaled

    def get_train_test_data(self):
        """Return all prepared data"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled,
            'X_test_scaled': self.X_test_scaled,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
