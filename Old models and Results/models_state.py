from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pandas as pd
import numpy as np

total_features = pd.read_pickle('total_features.parquet')
y = np.array(total_features.state.values, dtype=np.float32)


def model_training(X, y, feature_name):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=40, min_samples_split=5, max_features='sqrt',
                                               random_state=42),
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=40), n_estimators=100,
                                       learning_rate=0.01, random_state=42),
        'Ridge': RidgeClassifier(alpha=0.05, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.001, max_depth=40,
                                                       min_samples_split=2, subsample=0.9, random_state=42),
        'Svc': SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=2000, max_depth=10, learning_rate=0.001, objective='binary:logistic',
                                 eval_metric='logloss')
        }
    accuracies = {'features': feature_name}
    for name, model in models.items():
        pipe = Pipeline([
            # ('selector', SelectKBest(score_func=f_classif, k=min(30, len(X[0])))),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        accuracies[name] = scores.mean()
    return accuracies


X1 = np.vstack(total_features['amplitudes'].apply(lambda x: x[:100:5]).values)
d1 = model_training(X1,y, 'amplitudes')

X2 = np.hstack([
    np.vstack(total_features['fourier_features'].values),
    np.vstack(total_features['hilbert_features'].values)
])
d2 = model_training(X2,y, 'fourier+hilbert')

X3 = np.hstack([
    np.vstack(total_features['CWT_freq_var'].values),
    np.vstack(total_features['CWT_time_var'].values)
])
d3 = model_training(X3,y, 'CWT')

X4 = np.vstack(total_features['mfcc_mean'].values)
d4 = model_training(X4,y, 'mfcc')

X6 = np.hstack([X1, X4])
d5 = model_training(X6,y, 'amplitudes+mfcc')

accuracies_state = pd.DataFrame([d1, d2, d3, d4, d5])
accuracies_state.to_csv('accuracies_state.csv')
