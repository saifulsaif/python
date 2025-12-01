import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    """Class for creating visualizations of Titanic data and model results"""

    def __init__(self, df, models=None, feature_columns=None):
        self.df = df
        self.models = models
        self.feature_columns = feature_columns
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_survival_distribution(self):
        """Plot survival distribution"""
        print("\nGenerating Survival Distribution chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        self.df['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Survived (0=No, 1=Yes)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xticklabels(['Did Not Survive', 'Survived'], rotation=0)

        # Pie chart
        survival_counts = self.df['Survived'].value_counts()
        colors = ['#e74c3c', '#2ecc71']
        axes[1].pie(survival_counts, labels=['Did Not Survive', 'Survived'],
                   autopct='%1.1f%%', startangle=90, colors=colors)
        axes[1].set_title('Survival Rate', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('survival_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: survival_distribution.png")
        plt.close()

    def plot_survival_by_gender(self):
        """Plot survival by gender"""
        print("\nGenerating Survival by Gender chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        sns.countplot(data=self.df, x='Sex', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Survival by Gender - Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Gender', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Survived', labels=['No', 'Yes'])

        # Percentage plot
        survival_by_sex = self.df.groupby('Sex')['Survived'].mean()
        survival_by_sex.plot(kind='bar', ax=axes[1], color=['#3498db', '#e91e63'])
        axes[1].set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Gender', fontsize=12)
        axes[1].set_ylabel('Survival Rate', fontsize=12)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].set_ylim([0, 1])

        # Add percentage labels
        for i, v in enumerate(survival_by_sex):
            axes[1].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('survival_by_gender.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: survival_by_gender.png")
        plt.close()

    def plot_survival_by_class(self):
        """Plot survival by passenger class"""
        print("\nGenerating Survival by Class chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        sns.countplot(data=self.df, x='Pclass', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Survival by Class - Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Passenger Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Survived', labels=['No', 'Yes'])

        # Percentage plot
        survival_by_class = self.df.groupby('Pclass')['Survived'].mean()
        survival_by_class.plot(kind='bar', ax=axes[1], color=['#9b59b6', '#3498db', '#e67e22'])
        axes[1].set_title('Survival Rate by Class', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Passenger Class', fontsize=12)
        axes[1].set_ylabel('Survival Rate', fontsize=12)
        axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
        axes[1].set_ylim([0, 1])

        # Add percentage labels
        for i, v in enumerate(survival_by_class):
            axes[1].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('survival_by_class.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: survival_by_class.png")
        plt.close()

    def plot_age_distribution(self):
        """Plot age distribution by survival"""
        print("\nGenerating Age Distribution chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        survived = self.df[self.df['Survived'] == 1]['Age']
        not_survived = self.df[self.df['Survived'] == 0]['Age']

        axes[0].hist([not_survived, survived], bins=20, label=['Did Not Survive', 'Survived'],
                    color=['#e74c3c', '#2ecc71'], alpha=0.7)
        axes[0].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Age', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()

        # Box plot
        sns.boxplot(data=self.df, x='Survived', y='Age', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
        axes[1].set_title('Age by Survival Status', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Survived (0=No, 1=Yes)', fontsize=12)
        axes[1].set_ylabel('Age', fontsize=12)
        axes[1].set_xticklabels(['Did Not Survive', 'Survived'])

        plt.tight_layout()
        plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: age_distribution.png")
        plt.close()

    def plot_fare_distribution(self):
        """Plot fare distribution by survival"""
        print("\nGenerating Fare Distribution chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        sns.boxplot(data=self.df, x='Survived', y='Fare', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Fare by Survival Status', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Survived (0=No, 1=Yes)', fontsize=12)
        axes[0].set_ylabel('Fare', fontsize=12)
        axes[0].set_xticklabels(['Did Not Survive', 'Survived'])
        axes[0].set_ylim([0, 300])

        # Violin plot
        sns.violinplot(data=self.df, x='Pclass', y='Fare', hue='Survived',
                      ax=axes[1], palette=['#e74c3c', '#2ecc71'], split=True)
        axes[1].set_title('Fare Distribution by Class and Survival', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Passenger Class', fontsize=12)
        axes[1].set_ylabel('Fare', fontsize=12)
        axes[1].set_ylim([0, 300])
        axes[1].legend(title='Survived', labels=['No', 'Yes'])

        plt.tight_layout()
        plt.savefig('fare_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: fare_distribution.png")
        plt.close()

    def plot_family_size(self):
        """Plot survival by family size"""
        print("\nGenerating Family Size chart...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        sns.countplot(data=self.df, x='FamilySize', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
        axes[0].set_title('Survival by Family Size - Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Family Size', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].legend(title='Survived', labels=['No', 'Yes'])

        # Survival rate by family size
        survival_by_family = self.df.groupby('FamilySize')['Survived'].mean()
        survival_by_family.plot(kind='bar', ax=axes[1], color='#16a085')
        axes[1].set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Family Size', fontsize=12)
        axes[1].set_ylabel('Survival Rate', fontsize=12)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('family_size.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: family_size.png")
        plt.close()

    def plot_model_comparison(self):
        """Plot model accuracy comparison"""
        if not self.models:
            print("\nSkipping model comparison - no models provided")
            return

        print("\nGenerating Model Comparison chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]

        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(model_names, accuracies, color=colors)

        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, i, f'{acc*100:.2f}%', va='center', fontweight='bold')

        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison.png")
        plt.close()

    def plot_feature_importance(self, best_model_name, best_model):
        """Plot feature importance for tree-based models"""
        if best_model_name not in ['Decision Tree', 'Random Forest']:
            print(f"\nSkipping feature importance - {best_model_name} doesn't support feature importance")
            return

        print("\nGenerating Feature Importance chart...")

        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_importance.png")
        plt.close()

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of features"""
        print("\nGenerating Correlation Heatmap...")

        # Select numeric columns
        numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        if 'Sex_numeric' in self.df.columns:
            numeric_cols.append('Sex_numeric')

        correlation = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: correlation_heatmap.png")
        plt.close()

    def generate_all_charts(self, best_model_name=None, best_model=None):
        """Generate all visualization charts"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        self.plot_survival_distribution()
        self.plot_survival_by_gender()
        self.plot_survival_by_class()
        self.plot_age_distribution()
        self.plot_fare_distribution()

        if 'FamilySize' in self.df.columns:
            self.plot_family_size()

        self.plot_correlation_heatmap()

        if self.models:
            self.plot_model_comparison()

        if best_model_name and best_model and self.feature_columns:
            self.plot_feature_importance(best_model_name, best_model)

        print("\n" + "="*80)
        print("ALL VISUALIZATIONS SAVED! ✓")
        print("="*80)
