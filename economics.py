"""src/economics.py - Analyse économique et ROI de la maintenance prédictive"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .config import ECONOMICS_CONFIG
from .utils import setup_logger

logger = setup_logger(__name__)


class EconomicsAnalyzer:
    """Analyse économique et calcul ROI."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration personnalisée (sinon utilise ECONOMICS_CONFIG)
        """
        if config is None:
            config = ECONOMICS_CONFIG
        
        self.cost_per_hour_downtime = config.get('cost_per_hour_downtime', 500.0)
        self.cost_preventive_maintenance = config.get('cost_preventive_maintenance', 200.0)
        self.cost_emergency_repair = config.get('cost_emergency_repair', 1500.0)
        self.ml_investment_per_machine = config.get('ml_investment_per_machine', 100.0)
        self.occurrences_per_year = config.get('occurrences_per_year', 2.0)
        self.downtime_hours_map = config.get('downtime_hours', {})
        self.logger = logger

    def estimate_downtime_hours(self, rul_days: float, model: str) -> float:
        """Estime la durée d'arrêt en heures."""
        default_hours = self.downtime_hours_map.get('default', 5.0)
        return self.downtime_hours_map.get(str(model), default_hours)

    def cost_unplanned_downtime(self, rul_days: float, model: str) -> float:
        """Coût d'un arrêt non-planifié."""
        hours = self.estimate_downtime_hours(rul_days, model)
        return hours * self.cost_per_hour_downtime + self.cost_emergency_repair

    def cost_planned_maintenance(self) -> float:
        """Coût d'une maintenance préventive."""
        return self.cost_preventive_maintenance

    def potential_savings(self, rul_days: float, model: str) -> float:
        """Économies potentielles."""
        cost_unplanned = self.cost_unplanned_downtime(rul_days, model)
        cost_planned = self.cost_planned_maintenance()
        return max(0.0, cost_unplanned - cost_planned)

    def roi_per_machine(self, rul_days: float, model: str) -> Dict:
        """ROI annuel pour une machine.
        
        Returns:
            Dict avec ROI, économies, etc.
        """
        savings_per_event = self.potential_savings(rul_days, model)
        annual_savings = savings_per_event * self.occurrences_per_year
        annual_investment = self.ml_investment_per_machine
        annual_net = annual_savings - annual_investment
        
        roi_pct = 0.0
        if annual_investment > 0:
            roi_pct = (annual_net / annual_investment) * 100.0
        
        payback_months = 12.0
        if annual_savings > 0:
            payback_months = (12.0 * annual_investment / annual_savings)
        
        return {
            'annual_savings': round(annual_savings, 2),
            'annual_investment': round(annual_investment, 2),
            'annual_net_benefit': round(annual_net, 2),
            'roi_percent': round(roi_pct, 1),
            'payback_months': round(payback_months, 1),
        }

    def fleet_economics(self, df: pd.DataFrame) -> Dict:
        """Analyse économique du parc complet."""
        total_savings = 0.0
        total_investment = 0.0
        roi_by_model = {}
        
        try:
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                avg_rul = model_df['time'].mean() if 'time' in model_df.columns else 100.0
                
                roi = self.roi_per_machine(avg_rul, str(model))
                n_machines = len(model_df)
                
                total_savings += roi['annual_savings'] * n_machines
                total_investment += roi['annual_investment'] * n_machines
                roi_by_model[str(model)] = roi
        except Exception as e:
            self.logger.warning(f"Erreur calcul économique: {e}")
        
        total_net = total_savings - total_investment
        total_roi = 0.0
        if total_investment > 0:
            total_roi = (total_net / total_investment) * 100.0
        
        return {
            'total_fleet_savings': round(total_savings, 2),
            'total_ml_investment': round(total_investment, 2),
            'total_net_benefit': round(total_net, 2),
            'total_roi_percent': round(total_roi, 1),
            'roi_by_model': roi_by_model,
            'fleet_size': len(df),
        }

    def machine_economic_profile(self, row: pd.Series, model: str) -> Dict:
        """Profil économique pour une machine."""
        rul = row.get('time', 100.0)
        health = row.get('global_health_score', 50.0)
        
        cost_unplanned = self.cost_unplanned_downtime(rul, model)
        cost_planned = self.cost_planned_maintenance()
        savings = self.potential_savings(rul, model)
        
        return {
            'cost_unplanned_downtime': round(cost_unplanned, 2),
            'cost_planned_maintenance': round(cost_planned, 2),
            'potential_savings': round(savings, 2),
            'savings_percent': round((savings / cost_unplanned * 100) if cost_unplanned > 0 else 0, 1),
            'health_score': round(health, 1),
            'priority': 'HAUTE' if savings > 2000 else 'MOYENNE' if savings > 1000 else 'BASSE'
        }


logger.info("✅ Module economics chargé avec succès")
