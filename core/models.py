import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
    
    def train_basic_models(self, features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        logger.info("Training basic models (RF, SVM)")
        
        X, y = self._prepare_data(features_df, labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(C=1.0, gamma='scale', probability=True, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': y_test,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        return results
    
    def train_advanced_models(self, features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        logger.info("Training advanced models with optimization")
        
        X, y = self._prepare_data(features_df, labels)
        
        # Select only the most informative features to reduce noise and overfitting
        selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Set up models with manually tuned hyperparameters based on experimentation
        models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'svm_optimized': SVC(
                C=10,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42
            )
        }
        
        results = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': y_test,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_selector': selector
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        return results
    
    def train_ensemble_models(self, features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        logger.info("Training ensemble models")
        
        X, y = self._prepare_data(features_df, labels)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Base models for ensemble
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
            ('svm', SVC(C=10, gamma='scale', probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=150, max_depth=15, random_state=42)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]
        
        # Voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # Neural network
        neural_net = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        models = {
            'voting_ensemble': voting_ensemble,
            'neural_network': neural_net
        }
        
        results = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'feature_selector': selector
                }
                
                logger.info(f"{name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        return results
    
    def hyperparameter_optimization(self, features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        logger.info("Performing hyperparameter optimization")
        
        X, y = self._prepare_data(features_df, labels)
        
        # Random Forest optimization
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(X, y)
        
        # SVM optimization
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True),
            svm_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        svm_grid.fit(X, y)
        
        results = {
            'rf_optimized': {
                'model': rf_grid.best_estimator_,
                'accuracy': rf_grid.best_score_,
                'best_params': rf_grid.best_params_,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            },
            'svm_optimized': {
                'model': svm_grid.best_estimator_,
                'accuracy': svm_grid.best_score_,
                'best_params': svm_grid.best_params_,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
        }
        
        logger.info(f"Best RF parameters: {rf_grid.best_params_}")
        logger.info(f"Best RF CV score: {rf_grid.best_score_:.4f}")
        logger.info(f"Best SVM parameters: {svm_grid.best_params_}")
        logger.info(f"Best SVM CV score: {svm_grid.best_score_:.4f}")
        
        return results
    
    def cross_validate_models(self, features_df: pd.DataFrame, labels: List[str], cv_folds: int = 10) -> Dict[str, float]:
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        X, y = self._prepare_data(features_df, labels)
        
        models = {
            'rf': RandomForestClassifier(n_estimators=200, random_state=42),
            'svm': SVC(C=10, gamma='scale', random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, random_state=42),
            'et': ExtraTreesClassifier(n_estimators=150, random_state=42)
        }
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"Cross-validating {name}...")
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_results[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            logger.info(f"{name} CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def _prepare_data(self, features_df: pd.DataFrame, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale features
        X = self.scaler.fit_transform(numeric_features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def save_model(self, model_info: Dict[str, Any], filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model_info
    
    def predict(self, model_info: Dict[str, Any], features_df: pd.DataFrame) -> List[str]:
        model = model_info['model']
        scaler = model_info['scaler']
        label_encoder = model_info['label_encoder']
        
        # Prepare features
        numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = scaler.transform(numeric_features)
        
        # Apply feature selection if present
        if 'feature_selector' in model_info:
            X_scaled = model_info['feature_selector'].transform(X_scaled)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Decode labels
        predictions = label_encoder.inverse_transform(y_pred)
        
        return predictions.tolist()
    
    def get_feature_importance(self, model_info: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Handle feature selection
            if 'feature_selector' in model_info:
                selected_indices = model_info['feature_selector'].get_support(indices=True)
                selected_names = [feature_names[i] for i in selected_indices]
            else:
                selected_names = feature_names
            
            importance_dict = dict(zip(selected_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def evaluate_model(self, model_info: Dict[str, Any], features_df: pd.DataFrame, labels: List[str]) -> Dict[str, Any]:
        predictions = self.predict(model_info, features_df)
        
        # Convert to numeric for sklearn
        y_true = self.label_encoder.fit_transform(labels)
        y_pred = self.label_encoder.transform(predictions)
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.label_encoder.classes_,
                                           output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': predictions
        }