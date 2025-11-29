"""src/data_prep.py - Préparation et enrichissement des données"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

# Imports locaux
from .config import REQUIRED_COLUMNS, DATA_FILES, FEATURE_COLUMNS
from .utils import (
    setup_logger,
    normalize_column_names,
    convert_column_types,
    fill_numeric_columns,
    validate_dataframe_columns,
    check_missing_values,
    calculate_health_scores,
    calculate_global_health_score,
    calculate_degradation_rate,
    calculate_risk_factor,
    calculate_interaction_features,
    classify_risk_level_series,
    get_summary_statistics,
    get_fleet_statistics,
)
from .survival_analysis import fit_kmf_by_model

logger = setup_logger(__name__)

class DataPreprocessor:
    """Préprocesseur de données avec validation complète."""
    
    def __init__(self, required_columns: List[str] = None):
        """
        Args:
            required_columns: Liste des colonnes requises
        """
        self.required_columns = required_columns or REQUIRED_COLUMNS
        self.logger = logger
        
    def load_and_validate(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Charge et valide les données avec rapport détaillé.
        
        Returns:
            (DataFrame ou None, rapport de validation)
        """
        try:
            df = self.load_data(file_path)
            validation_report = self.comprehensive_validation(df)
            
            # Vérifier les colonnes manquantes
            if validation_report['missing_columns']:
                self.logger.error(
                    f"Colonnes manquantes: {validation_report['missing_columns']}"
                )
                return None, validation_report
            
            self.logger.info(f"✅ Données validées: {df.shape[0]} machines")
            return df, validation_report
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement: {e}")
            return None, {'error': str(e)}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Charge et normalise les données brutes."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        self.logger.info(f"📂 Chargement de {file_path}...")
        df = pd.read_csv(file_path)
        
        # Normalisation des noms de colonnes
        df = normalize_column_names(df)
        
        # Conversion des types de données
        type_conversions = {
            'machineid': 'int32',
            'time': 'float32',
            'event': 'int8',
            'volt': 'float32',
            'rotate': 'float32',
            'pressure': 'float32',
            'vibration': 'float32',
            'error_count': 'int16',
            'maint_count': 'int16',
            'age': 'int16'
        }
        df = convert_column_types(df, type_conversions)
        
        self.logger.info(f"✅ {len(df)} lignes chargées")
        return df

    def comprehensive_validation(self, df: pd.DataFrame) -> Dict:
        """Validation complète des données."""
        report = {
            'shape': df.shape,
            'missing_columns': [c for c in self.required_columns if c not in df.columns],
            'missing_values': check_missing_values(df),
            'data_types': validate_dataframe_columns(df, self.required_columns)[1],
        }
        
        # Statistiques de survie
        if 'time' in df.columns and 'event' in df.columns:
            failed = df[df['event'] == 1]
            censored = df[df['event'] == 0]
            report['survival_consistency'] = {
                'total_events': int(df['event'].sum()),
                'total_censored': len(censored),
                'avg_time_failure': float(failed['time'].mean()) if len(failed) > 0 else 0,
                'avg_time_censored': float(censored['time'].mean()) if len(censored) > 0 else 0,
            }
        
        return report

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée toutes les features avancées."""
        df = df.copy()
        
        # 1. Remplissage des valeurs manquantes
        numeric_cols = ['vibration', 'pressure', 'volt', 'rotate', 'error_count']
        df = fill_numeric_columns(df, numeric_cols, method='median')
        
        # 2. Scores de santé par composant
        df = calculate_health_scores(df)
        
        # 3. Score de santé global
        df['global_health_score'] = calculate_global_health_score(df)
        
        # 4. Features de dégradation
        df['degradation_rate'] = calculate_degradation_rate(df)
        df['risk_factor'] = calculate_risk_factor(df)
        
        # 5. Features d'interaction
        df = calculate_interaction_features(df)
        
        # 6. Classification du risque
        df['risk_level'] = classify_risk_level_series(df)

        # 7. Survival probabilities (Kaplan-Meier) par modèle
        try:
            kmf_dict = fit_kmf_by_model(df, time_col='time', event_col='event')
            # compute survival at fixed horizons
            horizons = [30, 90, 180]
            for h in horizons:
                col = f'survival_at_{h}'
                df[col] = df['model'].apply(lambda m: float(kmf_dict[m].predict(h)) if (m in kmf_dict and hasattr(kmf_dict[m], 'predict')) else float('nan'))

            # survival probability at the machine's observed time
            def _surv_at_time(row):
                m = row.get('model')
                t = row.get('time', None)
                if m in kmf_dict and t is not None:
                    try:
                        return float(kmf_dict[m].predict(float(t)))
                    except Exception:
                        return float('nan')
                return float('nan')

            df['survival_at_time'] = df.apply(_surv_at_time, axis=1)
        except Exception:
            # si survie impossible, remplir NaN
            df['survival_at_30'] = np.nan
            df['survival_at_90'] = np.nan
            df['survival_at_180'] = np.nan
            df['survival_at_time'] = np.nan

        # 8. Combiner risque heuristique + survie (escalade si faible prob survie)
        try:
            df['combined_risk'] = df['risk_level']
            # si prob survie à 30 jours < 0.5 alors escalade du niveau
            def _escalate(row):
                base = row['risk_level']
                surv30 = row.get('survival_at_30', np.nan)
                if pd.notna(surv30):
                    if surv30 < 0.3:
                        return 'CRITIQUE'
                    if surv30 < 0.6 and base not in ('CRITIQUE', 'ÉLEVÉ'):
                        return 'ÉLEVÉ'
                return base

            df['combined_risk'] = df.apply(_escalate, axis=1)
        except Exception:
            df['combined_risk'] = df['risk_level']
        
        self.logger.info(f"✅ {len(df)} lignes enrichies avec features avancées")
        return df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Retourne les statistiques du dataset."""
        return {
            'summary': get_summary_statistics(df),
            'fleet': get_fleet_statistics(df),
        }


# ==================== FONCTIONS PUBLIQUES ====================

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Charge les données depuis un fichier CSV.
    
    Args:
        file_path: Chemin du fichier CSV
    
    Returns:
        DataFrame ou None si erreur
    """
    try:
        return DataPreprocessor().load_data(file_path)
    except Exception as e:
        logger.error(f"Erreur chargement: {e}")
        return None

def validate_schema(df: pd.DataFrame, required: List[str] = None) -> List[str]:
    """Valide la présence des colonnes requises.
    
    Args:
        df: DataFrame à valider
        required: Liste des colonnes requises
    
    Returns:
        Liste des colonnes manquantes (vide si OK)
    """
    if required is None:
        required = REQUIRED_COLUMNS
    return [c for c in required if c not in df.columns]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features avancées.
    
    Args:
        df: DataFrame brut
    
    Returns:
        DataFrame enrichi
    """
    return DataPreprocessor().create_advanced_features(df)

def load_and_prepare(file_path: str) -> Optional[pd.DataFrame]:
    """Charge et prépare complètement les données (wrapper pratique).
    
    Args:
        file_path: Chemin du fichier
    
    Returns:
        DataFrame préparé ou None
    """
    preprocessor = DataPreprocessor()
    df, report = preprocessor.load_and_validate(file_path)
    
    if df is None:
        return None
    
    # Ajouter les features
    df = preprocessor.create_advanced_features(df)
    return df

logger.info("✅ Module data_prep chargé avec succès")