import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# 1. Load dataset
def load_data():
    url = 'https://raw.githubusercontent.com/emiliojuni0r/penambangan-data-ppd/refs/heads/main/StressLevelDataset.csv'
    df = pd.read_csv(url)
    return df

# 2. Train and save the tuned model
def train_and_save_model():
    print("Loading dataset...")
    df = load_data()
    
    # Memisahkan fitur dan target
    X = df.drop(['blood_pressure', 'stress_level'], axis=1)
    y = df['stress_level']
    
    # Split data (disamakan dengan random_state sebelumnya agar konsisten)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Menggunakan parameter hasil Hypertune Anda
    # C = 0.000774263682681127, solver = 'saga', penalty = 'l2'
    best_params = {
        'C': 0.000774263682681127,
        'penalty': 'l2',
        'solver': 'saga',
        'max_iter': 1000,
        'random_state': 42
    }
    
    print("\nTraining Logistic Regression with Tuned Parameters...")
    model = LogisticRegression(**best_params)
    
    # Cross validation untuk melihat stabilitas model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train pada full training set
    model.fit(X_train, y_train)
    
    # Evaluasi pada test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Test Accuracy   : {accuracy:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 3. Save the model
    with open('stress_level_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_names = list(X.columns)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("-" * 30)
    print("Model saved as 'stress_level_model.pkl'")
    print("Feature names saved as 'feature_names.pkl'")
    
    return model

if __name__ == "__main__":
    train_and_save_model()

# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression, Perceptron
# import warnings
# warnings.filterwarnings('ignore')

# # Load dataset
# def load_data():
#     url = 'https://raw.githubusercontent.com/emiliojuni0r/penambangan-data-ppd/refs/heads/main/StressLevelDataset.csv'
#     df = pd.read_csv(url)
#     return df

# # Train and save the best model
# def train_and_save_model():
#     print("Loading dataset...")
#     df = load_data()
    
#     # Memisahkan fitur dan target
#     X = df.drop('stress_level', axis=1)
#     y = df['stress_level']
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Model yang akan diuji
#     models = {
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
#         'AdaBoost': AdaBoostClassifier(random_state=42),
#         'K-Nearest Neighbors': KNeighborsClassifier(),
#         'Decision Tree': DecisionTreeClassifier(random_state=42),
#         'Naive Bayes': GaussianNB(),
#         'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#         'Perceptron': Perceptron(random_state=42)
#     }
    
#     best_model = None
#     best_score = 0
#     best_model_name = ""
    
#     print("\nTraining models...")
#     for name, model in models.items():
#         print(f"Training {name}...")
        
#         # Cross validation
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
#         # Train on full training set
#         model.fit(X_train, y_train)
        
#         # Evaluate on test set
#         y_pred = model.predict(X_test.values)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print(f"  {name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
#         print(f"  {name} - Test Accuracy: {accuracy:.4f}")
#         print(f"  {name} - F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
#         print("-" * 50)
        
#         if accuracy > best_score:
#             best_score = accuracy
#             best_model = model
#             best_model_name = name
    
#     print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    
#     # Save the best model
#     with open('stress_level_model.pkl', 'wb') as f:
#         pickle.dump(best_model, f)
    
#     # Save feature names
#     feature_names = list(X.columns)
#     with open('feature_names.pkl', 'wb') as f:
#         pickle.dump(feature_names, f)
    
#     print(f"\nModel saved as 'stress_level_model.pkl'")
#     print(f"Feature names saved as 'feature_names.pkl'")
    
#     return best_model, best_model_name, best_score

# if __name__ == "__main__":
#     train_and_save_model()