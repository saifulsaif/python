from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ModelTrainer:
    """Class for training multiple machine learning models"""

    def __init__(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.models = {}

    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n1. Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        y_pred_lr = lr.predict(self.X_test_scaled)
        acc_lr = accuracy_score(self.y_test, y_pred_lr)

        self.models['Logistic Regression'] = {
            'model': lr,
            'accuracy': acc_lr,
            'predictions': y_pred_lr,
            'scaled': True
        }

        print(f"   Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")
        return lr, acc_lr

    def train_decision_tree(self):
        """Train Decision Tree model"""
        print("\n2. Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt.fit(self.X_train, self.y_train)
        y_pred_dt = dt.predict(self.X_test)
        acc_dt = accuracy_score(self.y_test, y_pred_dt)

        self.models['Decision Tree'] = {
            'model': dt,
            'accuracy': acc_dt,
            'predictions': y_pred_dt,
            'scaled': False
        }

        print(f"   Accuracy: {acc_dt:.4f} ({acc_dt*100:.2f}%)")
        return dt, acc_dt

    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n3. Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(self.X_train, self.y_train)
        y_pred_rf = rf.predict(self.X_test)
        acc_rf = accuracy_score(self.y_test, y_pred_rf)

        self.models['Random Forest'] = {
            'model': rf,
            'accuracy': acc_rf,
            'predictions': y_pred_rf,
            'scaled': False
        }

        print(f"   Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
        return rf, acc_rf

    def train_all_models(self):
        """Train all models"""
        print("\n[STEP 9] TRAINING MODELS")
        print("-"*80)

        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()

        return self.models

    def get_models(self):
        """Return trained models dictionary"""
        return self.models

    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = max(self.models, key=lambda x: self.models[x]['accuracy'])
        best_model_info = self.models[best_model_name]
        return best_model_name, best_model_info
