"""Package src - Modules pour maintenance prédictive

Modules:
- config: Configuration centralisée
- utils: Fonctions utilitaires
- data_prep: Chargement et préparation des données
- models: Entraînement des modèles ML
- survival_analysis: Analyse de survie Kaplan-Meier
- alerting: Système d'alertes avancé
- economics: Analyse économique et ROI
- llm_report: Génération de rapports avec LLM
"""

from . import config
from . import utils
from . import data_prep
from . import models
from . import survival_analysis
from . import alerting
from . import economics

# Fonctions principales
from .config import get_config, get_data_path, get_models_dir
from .data_prep import load_data, create_features, validate_schema, load_and_prepare
from .models import AdvancedModelTrainer, save_model, load_model
from .survival_analysis import AdvancedSurvivalAnalyzer, fit_kmf_by_model, compare_models
from .alerting import AdvancedAlertingSystem
from .economics import EconomicsAnalyzer
from .utils import setup_logger, validate_dataframe_columns, classify_risk_level_series

__version__ = "1.0.0"
__all__ = [
    "config",
    "utils",
    "data_prep",
    "models",
    "survival_analysis",
    "alerting",
    "economics",
    # Classes
    "AdvancedModelTrainer",
    "AdvancedSurvivalAnalyzer",
    "AdvancedAlertingSystem",
    "EconomicsAnalyzer",
    # Fonctions
    "get_config",
    "get_data_path",
    "get_models_dir",
    "load_data",
    "create_features",
    "validate_schema",
    "load_and_prepare",
    "save_model",
    "load_model",
    "fit_kmf_by_model",
    "compare_models",
    "setup_logger",
]
