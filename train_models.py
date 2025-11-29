"""Script simple pour entraîner et sauvegarder les modèles à partir de Predictive_Table.csv
Usage:
    PYTHONPATH=. python3 src/train_models.py
"""
import json
import logging
from pathlib import Path

try:
    # when run as module from project root
    from src.models import AdvancedModelTrainer
    from src.config import get_data_path, get_models_dir
except Exception:
    # when executed from inside src/
    from models import AdvancedModelTrainer
    from config import get_data_path, get_models_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    # 1. localiser les données
    try:
        data_path = get_data_path('predictive')
    except Exception:
        data_path = str(Path(__file__).parent.parent / 'data' / 'Predictive_Table.csv')

    logger.info(f"Chargement des données depuis: {data_path}")

    # 2. charger et préparer les données via data_prep
    try:
        from .data_prep import load_and_prepare
    except Exception as e:
        logger.error("Impossible d'importer load_and_prepare: %s", e)
        return

    df = load_and_prepare(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # 3. entraîner
    trainer = AdvancedModelTrainer()

    # Classifier (prédire event)
    try:
        clf, clf_report = trainer.train_classifier(df)
        logger.info("Classifier trained. AUC: %s", clf_report.get('auc_roc'))
    except Exception as e:
        logger.exception("Erreur entraînement classifier: %s", e)
        clf = None
        clf_report = {}

    # Regressor (prédire time / RUL)
    try:
        reg, reg_report = trainer.train_rul_regressor(df)
        logger.info("Regressor trained. R2: %s", reg_report.get('r2'))
    except Exception as e:
        logger.exception("Erreur entraînement regressor: %s", e)
        reg = None
        reg_report = {}

    # 4. sauvegarder modèles + rapports
    models_dir = get_models_dir()
    to_save = {}
    if clf is not None:
        to_save['classifier'] = clf
    if reg is not None:
        to_save['regressor'] = reg

    if to_save:
        success = trainer.save_models(to_save, models_dir)
        if success:
            logger.info("Modèles sauvegardés dans %s", models_dir)

    # sauvegarder rapport
    report = {
        'classifier': clf_report,
        'regressor': reg_report,
    }
    report_path = Path(models_dir) / 'training_report.json'
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Rapport d'entraînement sauvegardé: %s", report_path)
    except Exception as e:
        logger.error("Impossible de sauvegarder le rapport: %s", e)


if __name__ == '__main__':
    main()
