"""src/models.py - Entraînement des modèles ML"""
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, mean_absolute_error, confusion_matrix, 
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

# Imports locaux
from .config import MODEL_CONFIG, get_model_path, get_models_dir
from .utils import setup_logger

logger = setup_logger(__name__)

class AdvancedModelTrainer:
    """Entraînement avancé des modèles avec optimisation."""
    
    def __init__(self, random_state: int = None, n_jobs: int = None):
        """
        Args:
            random_state: Seed aléatoire
            n_jobs: Nombre de processeurs (-1 pour tous)
        """
        self.random_state = random_state or MODEL_CONFIG['random_state']
        self.n_jobs = n_jobs or MODEL_CONFIG['n_jobs']
        self.feature_importance = {}
        self.model_performance = {}
        self.best_models = {}
        self.logger = logger
        
    def get_optimal_features(self, df: pd.DataFrame, target: str = 'event') -> List[str]:
        """Sélection automatique des meilleures features."""
        base_features = ['global_health_score', 'vibration', 'error_count', 'degradation_rate']
        
        # Vérifier les features disponibles
        available_features = [f for f in base_features if f in df.columns]
        
        # Features additionnelles
        additional_features = ['pressure', 'volt', 'rotate', 'age', 'maint_count', 'risk_factor']
        available_features.extend([f for f in additional_features if f in df.columns and f not in available_features])
        
        self.logger.debug(f"Features sélectionnées: {available_features}")
        return available_features
    
    def train_classifier(self, df: pd.DataFrame, features: List[str] = None,
                        target: str = 'event', test_size: float = None,
                        use_grid_search: bool = None) -> Tuple[RandomForestClassifier, Dict]:
        """Entraînement du classifieur de risque.
        
        Args:
            df: DataFrame
            features: Liste des colonnes d'entrée
            target: Colonne cible
            test_size: Proportion test/train
            use_grid_search: Utiliser GridSearchCV
        
        Returns:
            (modèle, rapport de performance)
        """
        if features is None:
            features = self.get_optimal_features(df, target)
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
        if use_grid_search is None:
            use_grid_search = MODEL_CONFIG['grid_search']
        
        # Préparation des données
        X = df[features].fillna(0)
        y = df[target]
        
        self.logger.info(f"Entraînement classificateur: {len(features)} features, {len(df)} samples")
        
        # Split stratifié
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Entraînement
        if use_grid_search:
            clf = self._train_classifier_gridsearch(X_train, y_train, features)
        else:
            clf = RandomForestClassifier(
                n_estimators=MODEL_CONFIG['n_estimators'],
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            clf.fit(X_train, y_train)
        
        # Évaluation
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        # Métriques
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        # roc_auc_score doesn't accept zero_division kwarg; wrap to handle errors
        try:
            report['auc_roc'] = float(roc_auc_score(y_test, y_pred_proba))
        except Exception:
            # si AUC impossible (ex: une seule classe), noter NaN
            report['auc_roc'] = float('nan')
        report['feature_importance'] = dict(zip(features, clf.feature_importances_.tolist()))
        report['n_samples_train'] = len(X_train)
        report['n_samples_test'] = len(X_test)
        
        self.model_performance['classifier'] = report
        self.feature_importance['classifier'] = report['feature_importance']
        self.best_models['classifier'] = clf
        
        self.logger.info(f"✅ Classificateur entraîné - AUC: {report['auc_roc']:.3f}")
        return clf, report
    
    def _train_classifier_gridsearch(self, X_train, y_train, features):
        """Entraînement avec GridSearchCV."""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        clf = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            clf, param_grid, cv=MODEL_CONFIG['cv_folds'], 
            scoring='f1', n_jobs=self.n_jobs, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        self.logger.debug(f"Meilleurs params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_rul_regressor(self, df: pd.DataFrame, features: List[str] = None,
                           target: str = 'time', test_size: float = None,
                           use_grid_search: bool = None) -> Tuple[RandomForestRegressor, Dict]:
        """Entraînement du régresseur RUL (Remaining Useful Life).
        
        Args:
            df: DataFrame
            features: Colonnes d'entrée
            target: Colonne cible (temps de vie)
            test_size: Proportion test/train
            use_grid_search: Utiliser GridSearchCV
        
        Returns:
            (modèle, rapport de performance)
        """
        if features is None:
            features = self.get_optimal_features(df, 'event')
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
        if use_grid_search is None:
            use_grid_search = MODEL_CONFIG['grid_search']
        
        # Préparation
        X = df[features].fillna(0)
        y = df[target]
        
        self.logger.info(f"Entraînement RUL regressor: {len(features)} features")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Entraînement
        if use_grid_search:
            regressor = self._train_regressor_gridsearch(X_train, y_train)
        else:
            regressor = RandomForestRegressor(
                n_estimators=MODEL_CONFIG['n_estimators'],
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            regressor.fit(X_train, y_train)
        
        # Prédictions et métriques
        y_pred = regressor.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': dict(zip(features, regressor.feature_importances_.tolist())),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
        }
        
        self.model_performance['regressor'] = metrics
        self.feature_importance['regressor'] = metrics['feature_importance']
        self.best_models['regressor'] = regressor
        
        self.logger.info(f"✅ RUL Regressor entraîné - R²: {r2:.3f}, MAE: {mae:.2f}")
        return regressor, metrics
    
    def _train_regressor_gridsearch(self, X_train, y_train):
        """Entraînement regressor avec GridSearchCV."""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
        }
        
        regressor = RandomForestRegressor(random_state=self.random_state)
        grid_search = GridSearchCV(
            regressor, param_grid, cv=MODEL_CONFIG['cv_folds'],
            scoring='neg_mean_absolute_error', n_jobs=self.n_jobs, verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def save_models(self, models_dict: Dict[str, any], models_dir: str = None) -> bool:
        """Sauvegarde les modèles.
        
        Args:
            models_dict: Dict avec {nom: modèle}
            models_dir: Répertoire de destination
        
        Returns:
            True si succès
        """
        if models_dir is None:
            models_dir = get_models_dir()
        
        try:
            os.makedirs(models_dir, exist_ok=True)
            
            for name, model in models_dict.items():
                path = os.path.join(models_dir, f"{name}.joblib")
                joblib.dump(model, path)
                self.logger.info(f"✅ Modèle {name} sauvegardé: {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde: {e}")
            return False

    def load_models(self, model_names: List[str], models_dir: str = None) -> Dict[str, any]:
        """Charge les modèles.
        
        Args:
            model_names: Liste des noms de modèles
            models_dir: Répertoire source
        
        Returns:
            Dict avec {nom: modèle}
        """
        if models_dir is None:
            models_dir = get_models_dir()
        
        loaded = {}
        for name in model_names:
            try:
                path = os.path.join(models_dir, f"{name}.joblib")
                if os.path.exists(path):
                    model = joblib.load(path)
                    loaded[name] = model
                    self.logger.info(f"✅ Modèle {name} chargé")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur chargement {name}: {e}")
        
        return loaded

    def get_performance_summary(self) -> Dict:
        """Retourne un résumé des performances."""
        return {
            'classifier': self.model_performance.get('classifier', {}),
            'regressor': self.model_performance.get('regressor', {}),
            'feature_importance': self.feature_importance,
        }


# ==================== FONCTIONS PUBLIQUES ====================

def save_model(model, path: str) -> bool:
    """Sauvegarde un modèle."""
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"✅ Modèle sauvegardé: {path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False

def load_model(path: str) -> Optional[any]:
    """Charge un modèle."""
    try:
        if os.path.exists(path):
            return joblib.load(path)
        logger.warning(f"⚠️  Fichier non trouvé: {path}")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return None

logger.info("✅ Module models chargé avec succès")