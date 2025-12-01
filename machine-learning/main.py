from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from visualizer import Visualizer


def main():
    """Main function to run the Titanic survival prediction pipeline"""

    print("="*80)
    print("TITANIC SURVIVAL PREDICTION MODEL")
    print("="*80)

    # STEP 1-3: Load and explore data
    data_loader = DataLoader('titanic_data.csv')
    df = data_loader.load_data()
    data_loader.explore_data()
    data_loader.analyze_data()

    # STEP 4-8: Preprocess and prepare data
    preprocessor = DataPreprocessor(df)
    df_processed = preprocessor.preprocess_data()
    preprocessor.feature_engineering()
    X, y = preprocessor.select_features()
    preprocessor.split_data(X, y)
    preprocessor.scale_features()

    # Get all prepared data
    train_test_data = preprocessor.get_train_test_data()

    # STEP 9: Train models
    trainer = ModelTrainer(
        X_train=train_test_data['X_train'],
        X_test=train_test_data['X_test'],
        y_train=train_test_data['y_train'],
        y_test=train_test_data['y_test'],
        X_train_scaled=train_test_data['X_train_scaled'],
        X_test_scaled=train_test_data['X_test_scaled']
    )
    models = trainer.train_all_models()

    # STEP 10-13: Evaluate models and make predictions
    evaluator = ModelEvaluator(
        models=models,
        y_test=train_test_data['y_test'],
        X=X,
        y=y,
        scaler=train_test_data['scaler'],
        feature_columns=train_test_data['feature_columns']
    )

    best_model_name, best_model_info = evaluator.evaluate_models()
    evaluator.detailed_report()
    cv_scores = evaluator.cross_validation()
    evaluator.show_feature_importance()
    evaluator.predict_new_passengers()

    # Generate visualizations
    visualizer = Visualizer(
        df=df_processed,
        models=models,
        feature_columns=train_test_data['feature_columns']
    )
    visualizer.generate_all_charts(
        best_model_name=best_model_name,
        best_model=best_model_info['model']
    )

    # Final summary
    print("\n" + "="*80)
    print("MODEL BUILDING COMPLETE! 🎉")
    print("="*80)

    print("\n📊 SUMMARY:")
    print("-"*80)
    print(f"✓ Dataset: {len(df)} passengers")
    print(f"✓ Features: {len(train_test_data['feature_columns'])}")
    print(f"✓ Models Trained: {len(models)}")
    print(f"✓ Best Model: {best_model_name}")
    print(f"✓ Test Accuracy: {best_model_info['accuracy']*100:.2f}%")
    print(f"✓ CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    print("\n🎯 KEY INSIGHTS:")
    print("-"*80)
    print(f"• Females had {df.groupby('Sex')['Survived'].mean()['female']*100:.1f}% survival rate")
    print(f"• Males had {df.groupby('Sex')['Survived'].mean()['male']*100:.1f}% survival rate")
    print(f"• 1st class: {df.groupby('Pclass')['Survived'].mean()[1]*100:.1f}% survival")
    print(f"• 3rd class: {df.groupby('Pclass')['Survived'].mean()[3]*100:.1f}% survival")


if __name__ == "__main__":
    main()
