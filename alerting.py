"""src/alerting.py - Système d'alertes avancé"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import os
import joblib

from .config import ALERT_LEVELS, MAINTENANCE_LEAD_TIME_DAYS
from .utils import setup_logger

logger = setup_logger(__name__)


class AdvancedAlertingSystem:
    """Système d'alertes prédictives avec analyse avancée."""
    
    def __init__(self, maintenance_lead_time_days: float = None,
                 classifier_path: str = None, rul_regressor_path: str = None):
        """
        Args:
            maintenance_lead_time_days: Délai d'anticipation pour la maintenance
            classifier_path: Chemin du classifieur ML
            rul_regressor_path: Chemin du régresseur RUL
        """
        self.maintenance_lead_time = maintenance_lead_time_days or MAINTENANCE_LEAD_TIME_DAYS
        self.alert_levels = ALERT_LEVELS
        self.classifier = None
        self.rul_regressor = None
        self.alert_history = []
        self.logger = logger
        
        self._load_models(classifier_path, rul_regressor_path)
    
    def _load_models(self, classifier_path: Optional[str], rul_regressor_path: Optional[str]):
        """Charge les modèles ML si disponibles."""
        if classifier_path and os.path.exists(classifier_path):
            try:
                self.classifier = joblib.load(classifier_path)
                self.logger.info(f"✅ Classificateur chargé: {classifier_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur chargement classificateur: {e}")
        
        if rul_regressor_path and os.path.exists(rul_regressor_path):
            try:
                self.rul_regressor = joblib.load(rul_regressor_path)
                self.logger.info(f"✅ RUL Regressor chargé: {rul_regressor_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur chargement RUL regressor: {e}")

    def calculate_risk_score(self, row: pd.Series) -> Dict:
        """Calcule un score de risque multi-dimensionnel."""
        risk_factors = {}
        
        # Facteur de santé
        health = row.get('global_health_score', 50)
        risk_factors['health_risk'] = max(0, min(1, (100 - health) / 100))
        
        # Facteur de dégradation
        degradation = row.get('degradation_rate', 0)
        risk_factors['degradation_risk'] = min(1, degradation / 10)
        
        # Facteur d'âge
        age = row.get('age', 0)
        risk_factors['age_risk'] = min(1, age / 20)
        
        # Facteur d'erreurs
        errors = row.get('error_count', 0)
        risk_factors['error_risk'] = min(1, errors / 50)
        
        # Score global pondéré
        weights = {'health_risk': 0.4, 'degradation_risk': 0.3, 'age_risk': 0.2, 'error_risk': 0.1}
        total_risk = sum(risk_factors[f] * w for f, w in weights.items())
        
        return {
            'total_risk': min(1.0, total_risk),
            'risk_factors': risk_factors,
        }

    def get_alert_level(self, rul_days: float) -> str:
        """Détermine le niveau d'alerte basé sur RUL."""
        for level in ['CRITIQUE', 'HAUTE', 'MOYENNE', 'BASSE']:
            if rul_days <= self.alert_levels[level]['rul_max']:
                return level
        return 'BASSE'

    def predict_failure_risk(self, row: pd.Series, feature_cols: List[str] = None) -> Tuple[float, float, Dict]:
        """Prédiction avancée avec analyse multi-facteurs."""
        if feature_cols is None:
            feature_cols = ['global_health_score', 'vibration', 'error_count', 'degradation_rate']
        
        prob_failure = self._predict_probability(row, feature_cols)
        rul_days = self._predict_rul(row, feature_cols)
        risk_analysis = self.calculate_risk_score(row)
        
        return prob_failure, rul_days, risk_analysis

    def _predict_probability(self, row: pd.Series, feature_cols: List[str]) -> float:
        """Prédit la probabilité de panne."""
        if self.classifier is not None:
            try:
                available = [f for f in feature_cols if f in row.index and pd.notna(row[f])]
                if len(available) > 0:
                    X = row[available].fillna(0).values.reshape(1, -1)
                    return float(self.classifier.predict_proba(X)[0, 1])
            except Exception as e:
                self.logger.debug(f"Erreur prédiction probabilité: {e}")
        
        health = row.get('global_health_score', 50)
        return max(0, min(1, (100 - health) / 100))

    def _predict_rul(self, row: pd.Series, feature_cols: List[str]) -> float:
        """Prédit le RUL (Remaining Useful Life)."""
        if self.rul_regressor is not None:
            try:
                available = [f for f in feature_cols if f in row.index and pd.notna(row[f])]
                if len(available) > 0:
                    X = row[available].fillna(0).values.reshape(1, -1)
                    return max(1, float(self.rul_regressor.predict(X)[0]))
            except Exception as e:
                self.logger.debug(f"Erreur prédiction RUL: {e}")
        
        return float(row.get('time', 30))

    def generate_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """Génère des alertes pour le parc."""
        alerts = []
        
        for idx, row in df.iterrows():
            try:
                machine_id = int(row.get('machineID', idx))
                prob_failure, rul, risk_analysis = self.predict_failure_risk(row)
                alert_level = self._determine_alert_level(rul, prob_failure, risk_analysis['total_risk'])
                
                if alert_level != 'BASSE':
                    alert = {
                        'machine_id': machine_id,
                        'model': str(row.get('model', 'Unknown')),
                        'alert_level': alert_level,
                        'rul_days': round(rul, 1),
                        'prob_failure': round(prob_failure, 3),
                        'risk_score': round(risk_analysis['total_risk'], 3),
                        'recommended_action': self.alert_levels[alert_level]['action'],
                        'maintenance_deadline': (datetime.now() + timedelta(days=rul - self.maintenance_lead_time)).strftime('%Y-%m-%d'),
                        'priority': self.alert_levels[alert_level]['priority'],
                        'health_score': round(float(row.get('global_health_score', 50)), 1),
                        'age': int(row.get('age', 0)),
                        'timestamp': datetime.now().isoformat(),
                    }
                    alerts.append(alert)
            except Exception as e:
                self.logger.debug(f"Erreur alerte machine {idx}: {e}")
        
        alerts.sort(key=lambda x: (x['priority'], -x['prob_failure']))
        self.alert_history.extend(alerts)
        self.logger.info(f"✅ {len(alerts)} alertes générées")
        return alerts

    def _determine_alert_level(self, rul: float, prob_failure: float, risk_score: float) -> str:
        """Détermine le niveau d'alerte."""
        base = self.get_alert_level(rul)
        
        if prob_failure > 0.8 or risk_score > 0.8:
            return 'CRITIQUE'
        elif prob_failure > 0.6 or risk_score > 0.6:
            if base in ['BASSE', 'MOYENNE']:
                return 'HAUTE'
        elif prob_failure > 0.4 and base == 'BASSE':
            return 'MOYENNE'
        
        return base

    def maintenance_calendar(self, alerts: List[Dict], weeks_ahead: int = 4) -> Dict:
        """Calendrier de maintenance optimisé."""
        calendar = {f'Semaine_{i+1}': [] for i in range(weeks_ahead)}
        
        for alert in alerts:
            days_until = alert['rul_days'] - self.maintenance_lead_time
            if 0 < days_until <= weeks_ahead * 7:
                week_num = min(weeks_ahead - 1, int(days_until // 7))
                week_key = f'Semaine_{week_num + 1}'
                calendar[week_key].append(alert)
        
        for week in calendar:
            calendar[week].sort(key=lambda x: x['priority'])
        
        return calendar

    def get_alert_statistics(self, alerts: List[Dict]) -> Dict:
        """Statistiques sur les alertes."""
        if not alerts:
            return {'total_alerts': 0, 'by_level': {}, 'by_model': {}}
        
        stats = {
            'total_alerts': len(alerts),
            'by_level': {},
            'by_model': {},
        }
        
        for level in self.alert_levels:
            count = len([a for a in alerts if a['alert_level'] == level])
            stats['by_level'][level] = count
        
        for alert in alerts:
            model = alert['model']
            stats['by_model'][model] = stats['by_model'].get(model, 0) + 1
        
        return stats


logger.info("✅ Module alerting chargé avec succès")
