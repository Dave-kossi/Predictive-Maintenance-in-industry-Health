"""src/utils.py - Fonctions utilitaires partagées"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configuration du logging
def setup_logger(name: str, log_level: str = 'INFO') -> logging.Logger:
    """Configure et retourne un logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger

# Logger principal du module
logger = setup_logger(__name__)

# ==================== VALIDATION ====================

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Valide la présence des colonnes requises.
    
    Returns:
        (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing

def validate_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Retourne les types de données du DataFrame."""
    return df.dtypes.to_dict()

def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Retourne le nombre de valeurs manquantes par colonne."""
    return df.isnull().sum().to_dict()

# ==================== NETTOYAGE DES DONNÉES ====================

def fill_numeric_columns(df: pd.DataFrame, columns: List[str], method: str = 'median') -> pd.DataFrame:
    """Remplit les valeurs manquantes dans les colonnes numériques.
    
    Args:
        df: DataFrame
        columns: Liste des colonnes à traiter
        method: 'median', 'mean', ou 'forward_fill'
    
    Returns:
        DataFrame avec valeurs remplies
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'forward_fill':
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df

def convert_column_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """Convertit les types de colonnes spécifiées.
    
    Args:
        df: DataFrame
        type_mapping: Dict comme {'col_name': 'int32', 'col2': 'float32'}
    
    Returns:
        DataFrame avec types convertis
    """
    df = df.copy()
    
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                logger.debug(f"✅ Colonne {col} convertie en {dtype}")
            except Exception as e:
                logger.warning(f"⚠️  Impossible de convertir {col} en {dtype}: {e}")
    
    return df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes en standardisant les clés utilisées par le code.

    - Conserve explicitement `machineID` (comme dans le CSV fourni).
    - Mappe les variantes courantes (e.g. `machineid`, `machine_id`) vers `machineID`.
    - Garde les autres noms en minuscules sans espaces.
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        raw = col.strip()
        key = raw.lower().replace(' ', '_')
        if key in ('machineid', 'machine_id'):
            new_cols.append('machineID')
        else:
            new_cols.append(key)

    df.columns = new_cols
    return df

# ==================== CALCULS DE FEATURES ====================

def calculate_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les scores de santé par composant.
    
    Returns:
        DataFrame avec colonnes de scores ajoutées
    """
    df = df.copy()
    
    # Vibration score
    if 'vibration' in df.columns:
        df['vibration_score'] = np.maximum(0, 100 - (df['vibration'] - 35) * 2)
    
    # Pressure score
    if 'pressure' in df.columns:
        df['pressure_score'] = np.maximum(0, 100 - np.abs(df['pressure'] - 100) * 2)
    
    # Error score
    if 'error_count' in df.columns:
        df['error_score'] = np.maximum(0, 100 - df['error_count'] * 2)
    
    # Maintenance score
    if 'maint_count' in df.columns:
        df['maintenance_score'] = np.minimum(df['maint_count'] * 3, 100)
    
    # Age score
    if 'age' in df.columns:
        df['age_score'] = np.maximum(0, 100 - df['age'] * 5)
    
    return df

def calculate_global_health_score(df: pd.DataFrame, 
                                   weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Calcule le score de santé global.
    
    Args:
        df: DataFrame contenant les scores de santé par composant
        weights: Dict avec poids pour chaque composant
    
    Returns:
        Series avec scores globaux
    """
    if weights is None:
        weights = {
            'vibration': 0.30,
            'pressure': 0.25,
            'error': 0.20,
            'maintenance': 0.15,
            'age': 0.10,
        }
    
    global_score = pd.Series(0.0, index=df.index)
    
    for component, weight in weights.items():
        col_name = f'{component}_score'
        if col_name in df.columns:
            global_score += df[col_name] * weight
    
    return global_score.clip(0, 100).round(2)

def calculate_degradation_rate(df: pd.DataFrame) -> pd.Series:
    """Calcule le taux de dégradation (vibration/âge)."""
    if 'vibration' in df.columns and 'age' in df.columns:
        return (df['vibration'] / (df['age'] + 1)).round(4)
    return pd.Series(0.0, index=df.index)

def calculate_risk_factor(df: pd.DataFrame) -> pd.Series:
    """Calcule le facteur de risque (santé × dégradation)."""
    if 'global_health_score' in df.columns and 'degradation_rate' in df.columns:
        return ((100 - df['global_health_score']) * df['degradation_rate']).round(2)
    return pd.Series(0.0, index=df.index)

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features d'interaction."""
    df = df.copy()
    
    # Vibration/Pressure ratio
    if 'vibration' in df.columns and 'pressure' in df.columns:
        df['vibration_pressure_ratio'] = (
            df['vibration'] / df['pressure']
        ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Error/Maintenance ratio
    if 'error_count' in df.columns and 'maint_count' in df.columns:
        df['error_maintenance_ratio'] = (
            df['error_count'] / (df['maint_count'] + 1)
        ).round(2)
    
    return df

# ==================== CLASSIFICATION DE RISQUE ====================

def classify_risk_level(health_score: float, vibration: float, error_count: int,
                        thresholds: Optional[Dict] = None) -> str:
    """Classifie le niveau de risque d'une machine.
    
    Args:
        health_score: Score de santé global (0-100)
        vibration: Valeur de vibration
        error_count: Nombre d'erreurs
        thresholds: Dict des seuils de risque
    
    Returns:
        'CRITIQUE', 'ÉLEVÉ', 'MODÉRÉ' ou 'FAIBLE'
    """
    if thresholds is None:
        thresholds = {
            'CRITIQUE': {'health': 30, 'vibration': 45, 'errors': 40},
            'ÉLEVÉ': {'health': 50, 'vibration': 40, 'errors': 30},
            'MODÉRÉ': {'health': 70, 'vibration': 50, 'errors': 50},
        }
    
    # Vérifier CRITIQUE d'abord (plus grave)
    crit = thresholds['CRITIQUE']
    if health_score < crit['health'] or vibration > crit['vibration'] or error_count > crit['errors']:
        return 'CRITIQUE'
    
    # Puis ÉLEVÉ
    elev = thresholds['ÉLEVÉ']
    if health_score < elev['health'] or vibration > elev['vibration'] or error_count > elev['errors']:
        return 'ÉLEVÉ'
    
    # Puis MODÉRÉ
    mod = thresholds['MODÉRÉ']
    if health_score < mod['health']:
        return 'MODÉRÉ'
    
    # Sinon FAIBLE
    return 'FAIBLE'

def classify_risk_level_series(df: pd.DataFrame, thresholds: Optional[Dict] = None) -> pd.Series:
    """Classifie le risque pour toutes les machines."""
    return df.apply(
        lambda row: classify_risk_level(
            row.get('global_health_score', 50),
            row.get('vibration', 0),
            int(row.get('error_count', 0)),
            thresholds
        ),
        axis=1
    )

# ==================== GESTION DES FICHIERS ====================

def ensure_directory_exists(directory: str) -> Path:
    """Crée un répertoire s'il n'existe pas."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_path(directory: str, filename: str) -> Path:
    """Retourne un chemin complet après avoir créé le répertoire."""
    ensure_directory_exists(directory)
    return Path(directory) / filename

# ==================== STATISTIQUES ====================

def get_summary_statistics(df: pd.DataFrame, numeric_only: bool = True) -> Dict:
    """Retourne les statistiques résumées du DataFrame."""
    return {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'dtypes': df.dtypes.to_dict(),
        'missing_count': df.isnull().sum().to_dict(),
        'description': df.describe(include='all').to_dict() if numeric_only else None,
    }

def get_fleet_statistics(df: pd.DataFrame) -> Dict:
    """Retourne des statistiques sur le parc de machines."""
    return {
        'total_machines': len(df),
        'unique_models': df['model'].nunique() if 'model' in df.columns else 0,
        'avg_age': df['age'].mean() if 'age' in df.columns else 0,
        'avg_health': df['global_health_score'].mean() if 'global_health_score' in df.columns else 0,
        'failure_rate': df['event'].mean() * 100 if 'event' in df.columns else 0,
        'machines_by_model': df['model'].value_counts().to_dict() if 'model' in df.columns else {},
    }

logger.info("✅ Module utils chargé avec succès")
