"""src/config.py - Configuration centralisée du projet"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Charger les variables d'environnement si disponible
    load_dotenv()
except Exception:
    # python-dotenv non installé — ignorer et continuer
    def load_dotenv():
        return None

# ==================== CHEMINS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Créer les répertoires s'ils n'existent pas
for dir_path in [MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ==================== FICHIERS DE DONNÉES ====================
DATA_FILES = {
    'predictive': DATA_DIR / "Predictive_Table.csv",
    'kaplan': DATA_DIR / "votre_table_kaplan.csv",
}

# ==================== COLONNES ====================
REQUIRED_COLUMNS = [
    'machineID', 'model', 'age', 'time', 'event', 'volt', 'rotate', 'pressure',
    'vibration', 'error_count', 'maint_count'
]

FEATURE_COLUMNS = {
    'health_indicators': ['vibration_score', 'pressure_score', 'error_score', 'maintenance_score', 'age_score'],
    'advanced': ['global_health_score', 'degradation_rate', 'risk_factor', 'vibration_pressure_ratio', 'error_maintenance_ratio'],
    'model_input': ['global_health_score', 'vibration', 'error_count', 'degradation_rate'],
    'clustering': ['vibration', 'pressure', 'error_count'],
}

# ==================== CONFIGURATION DU MODÈLE ====================
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'n_estimators': 100,
    'n_clusters': 3,
    'cv_folds': 5,
    'grid_search': True,
    'n_jobs': -1,  # Utiliser tous les processeurs
}

# ==================== CONFIGURATION DU SCORING ====================
SCORING_THRESHOLDS = {
    'CRITIQUE': {'health': 30, 'vibration': 45, 'errors': 40},
    'ÉLEVÉ': {'health': 50, 'vibration': 40, 'errors': 30},
    'MODÉRÉ': {'health': 70, 'vibration': 50, 'errors': 50},
    'FAIBLE': {'health': 100, 'vibration': 100, 'errors': 100},
}

# Coefficients de poids pour le score de santé
HEALTH_SCORE_WEIGHTS = {
    'vibration': 0.30,
    'pressure': 0.25,
    'error': 0.20,
    'maintenance': 0.15,
    'age': 0.10,
}

# ==================== CONFIGURATION DES ALERTES ====================
ALERT_LEVELS = {
    'CRITIQUE': {'color': '#d32f2f', 'rul_max': 7, 'action': 'IMMÉDIAT', 'priority': 0},
    'HAUTE': {'color': '#f57c00', 'rul_max': 14, 'action': 'CETTE_SEMAINE', 'priority': 1},
    'MOYENNE': {'color': '#fbc02d', 'rul_max': 30, 'action': 'CE_MOIS', 'priority': 2},
    'BASSE': {'color': '#388e3c', 'rul_max': 365, 'action': 'SURVEILLANCE', 'priority': 3}
}

MAINTENANCE_LEAD_TIME_DAYS = 2.0

# ==================== CONFIGURATION ÉCONOMIQUE ====================
ECONOMICS_CONFIG = {
    'cost_per_hour_downtime': 500.0,  # € par heure d'arrêt
    'cost_preventive_maintenance': 200.0,  # € par maintenance planifiée
    'cost_emergency_repair': 1500.0,  # € surcoût réparation urgente
    'ml_investment_per_machine': 100.0,  # € investissement ML par machine
    'occurrences_per_year': 2.0,  # Nombre de défaillances/an si pas de prédiction
    'downtime_hours': {  # Par modèle de machine
        'model1': 4.0,
        'model2': 6.0,
        'model3': 5.0,
        'model4': 3.5,
        'default': 5.0,
    }
}

# ==================== CONFIGURATION LOGGING ====================
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# ==================== CONFIGURATION STREAMLIT ====================
STREAMLIT_CONFIG = {
    'page_title': 'Maintenance Prédictive 🔧',
    'page_icon': '🔧',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# ==================== CONFIGURATION LLM ====================
LLM_CONFIG = {
    'provider': os.getenv('LLM_PROVIDER', 'huggingface'),
    'api_token': os.getenv('HF_API_TOKEN', None),
    'model': os.getenv('LLM_MODEL', 'mistralai/Mistral-7B-Instruct'),
    'timeout': 30,  # secondes
    'enabled': bool(os.getenv('HF_API_TOKEN')),
}

# ==================== CONFIGURATION RAPPORTS ====================
REPORT_CONFIG = {
    'formats': ['json', 'html', 'pdf'],
    'default_format': 'json',
    'include_charts': True,
    'output_dir': PROJECT_ROOT / 'reports',
}

# Créer le répertoire rapports
REPORT_CONFIG['output_dir'].mkdir(exist_ok=True)

# ==================== FONCTIONS D'ACCÈS ====================

def get_config():
    """Retourne toute la configuration sous forme de dictionnaire."""
    return {
        'PROJECT_ROOT': str(PROJECT_ROOT),
        'DATA_DIR': str(DATA_DIR),
        'MODELS_DIR': str(MODELS_DIR),
        'LOGS_DIR': str(LOGS_DIR),
        'DATA_FILES': {k: str(v) for k, v in DATA_FILES.items()},
        'REQUIRED_COLUMNS': REQUIRED_COLUMNS,
        'FEATURE_COLUMNS': FEATURE_COLUMNS,
        'MODEL_CONFIG': MODEL_CONFIG,
        'SCORING_THRESHOLDS': SCORING_THRESHOLDS,
        'HEALTH_SCORE_WEIGHTS': HEALTH_SCORE_WEIGHTS,
        'ALERT_LEVELS': ALERT_LEVELS,
        'ECONOMICS_CONFIG': ECONOMICS_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG,
        'STREAMLIT_CONFIG': STREAMLIT_CONFIG,
        'LLM_CONFIG': LLM_CONFIG,
        'REPORT_CONFIG': REPORT_CONFIG,
    }

def get_data_path(key='predictive'):
    """Retourne le chemin d'un fichier de données."""
    path = DATA_FILES.get(key)
    if path and path.exists():
        return str(path)
    raise FileNotFoundError(f"Fichier de données '{key}' non trouvé: {path}")

def get_models_dir():
    """Retourne le chemin du répertoire des modèles."""
    return str(MODELS_DIR)

def get_model_path(model_name):
    """Retourne le chemin d'un modèle sauvegardé."""
    return str(MODELS_DIR / f"{model_name}.joblib")

def ensure_dirs():
    """Crée tous les répertoires nécessaires."""
    for dir_path in [MODELS_DIR, LOGS_DIR, REPORT_CONFIG['output_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Assurer que les répertoires existent au chargement
ensure_dirs()

# ==================== MAPPAGE / LÉGENDE DES MODÈLES ====================
# Mappe les codes techniques ('model1', etc.) vers des labels courts A/B/C
# Choix par défaut: regrouper les modèles moins critiques sous 'C'.
MODEL_LABEL_MAP = {
    'model2': 'A',  # plus critique (pannes fréquentes / court time)
    'model1': 'B',  # critique élevée
    'model3': 'C',  # moins critique
    'model4': 'C',  # moins critique
}

# Description lisible pour la légende (ordre affiché = A, B, C)
MODEL_LABEL_DESC = [
    ('A', 'Modèle A — Correspond à: model2 — Priorité haute, pannes fréquentes.'),
    ('B', 'Modèle B — Correspond à: model1 — Priorité moyenne-élevée.'),
    ('C', 'Modèle C — Correspond à: model3, model4 — Priorité moindre.'),
]

def map_model_to_label(code):
    """Retourne le label A/B/C pour un code de modèle technique."""
    return MODEL_LABEL_MAP.get(str(code).lower(), 'C')

