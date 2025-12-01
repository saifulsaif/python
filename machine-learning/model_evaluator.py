import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    """Class for evaluating models and making predictions"""

    def __init__(self, models, y_test, X, y, scaler, feature_columns):
        self.models = models
        self.y_test = y_test
        self.X = X
        self.y = y
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.best_model_name = None
        self.best_model_info = None

    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n[STEP 10] EVALUATING MODELS")
        print("-"*80)

        # Find best model
        self.best_model_name = max(self.models, key=lambda x: self.models[x]['accuracy'])
        self.best_model_info = self.models[self.best_model_name]

        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Accuracy: {self.best_model_info['accuracy']*100:.2f}%")

        # Model comparison
        print("\nMODEL COMPARISON:")
        print("-"*60)
        print(f"{'Model':<25} {'Accuracy':<15} {'Needs Scaling'}")
        print("-"*60)

        for name, info in self.models.items():
            marker = "🏆" if name == self.best_model_name else "  "
            scaling = "Yes" if info['scaled'] else "No"
            print(f"{marker} {name:<23} {info['accuracy']:.4f} ({info['accuracy']*100:.1f}%)  {scaling}")

        return self.best_model_name, self.best_model_info

    def detailed_report(self):
        """Generate detailed classification report"""
        print(f"\nDETAILED REPORT: {self.best_model_name}")
        print("-"*60)
        print(classification_report(self.y_test, self.best_model_info['predictions'],
                                  target_names=['Did Not Survive', 'Survived']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.best_model_info['predictions'])
        print("Confusion Matrix:")
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")

    def cross_validation(self):
        """Perform cross-validation on best model"""
        print("\n[STEP 11] CROSS-VALIDATION")
        print("-"*80)

        best_model = self.best_model_info['model']

        if self.best_model_info['scaled']:
            X_for_cv = self.scaler.fit_transform(self.X)
        else:
            X_for_cv = self.X

        cv_scores = cross_val_score(best_model, X_for_cv, self.y, cv=5, scoring='accuracy')

        print("5-Fold Cross-Validation Scores:", cv_scores)
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        return cv_scores

    def show_feature_importance(self):
        """Show feature importance for tree-based models"""
        if self.best_model_name in ['Decision Tree', 'Random Forest']:
            print("\n[STEP 12] FEATURE IMPORTANCE")
            print("-"*80)

            best_model = self.best_model_info['model']

            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\nTop Features:")
            for idx, row in importance_df.iterrows():
                bar = '█' * int(row['Importance'] * 50)
                print(f"{row['Feature']:<20} {row['Importance']:.4f}  {bar}")

    def predict_new_passengers(self):
        """Make predictions on new passenger data"""
        print("\n[STEP 13] PREDICTIONS ON NEW PASSENGERS")
        print("-"*80)

        # Create new passenger data
        new_passengers = pd.DataFrame({
            'Pclass': [3, 1, 2],
            'Sex_numeric': [0, 1, 0],  # 0=male, 1=female
            'Age': [25, 35, 5],
            'SibSp': [0, 1, 2],
            'Parch': [0, 0, 2],
            'Fare': [8.0, 100.0, 20.0],
            'Embarked_C': [0, 1, 0],
            'Embarked_Q': [0, 0, 1],
            'Embarked_S': [1, 0, 0],
            'FamilySize': [1, 2, 5],
            'IsAlone': [1, 0, 0],
            'FarePerPerson': [8.0, 50.0, 4.0]
        })

        print("\nNew Passengers:")
        descriptions = [
            "25-year-old male, 3rd class, alone, $8 fare",
            "35-year-old female, 1st class, with spouse, $100 fare",
            "5-year-old male, 2nd class, with family, $20 fare"
        ]

        for i, desc in enumerate(descriptions, 1):
            print(f"{i}. {desc}")

        # Prepare data for prediction
        best_model = self.best_model_info['model']

        if self.best_model_info['scaled']:
            new_passengers_processed = self.scaler.transform(new_passengers)
        else:
            new_passengers_processed = new_passengers

        # Make predictions
        predictions = best_model.predict(new_passengers_processed)
        probabilities = best_model.predict_proba(new_passengers_processed)

        print("\nPREDICTION RESULTS:")
        print("-"*60)

        for i, (desc, pred, prob) in enumerate(zip(descriptions, predictions, probabilities), 1):
            status = "✓ SURVIVED" if pred == 1 else "✗ DID NOT SURVIVE"
            confidence = prob[pred] * 100
            survival_prob = prob[1] * 100

            print(f"\nPassenger {i}: {desc}")
            print(f"  Prediction: {status}")
            print(f"  Survival Probability: {survival_prob:.1f}%")
            print(f"  Confidence: {confidence:.1f}%")

        return predictions, probabilities

    def get_best_model_info(self):
        """Return best model information"""
        return self.best_model_name, self.best_model_info
