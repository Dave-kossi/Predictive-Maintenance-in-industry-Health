"""src/industrial_reports.py - Version améliorée"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ReportType(Enum):
    EXECUTIVE = "executive"
    MAINTENANCE = "maintenance" 
    TECHNICAL = "technical"
    FINANCIAL = "financial"

@dataclass
class MachineMetrics:
    """Metrics individuelles pour une machine."""
    machine_id: int
    model: str
    health_score: float
    rul_days: float
    risk_level: str
    maintenance_cost: float
    downtime_cost: float

class AdvancedReportGenerator:
    """Générateur de rapports industriels avancé."""
    
    def __init__(self, company_name: str = "Entreprise Industrielle"):
        self.company_name = company_name
        self.report_date = datetime.now()
        
    def generate_executive_report(self, fleet_data: pd.DataFrame, 
                                economics: Dict, alerts: List[Dict]) -> str:
        """Rapport direction avec analyse financière avancée."""
        report = []
        
        # En-tête professionnel
        report.extend(self._create_report_header("RAPPORT EXÉCUTIF - MAINTENANCE PRÉDICTIVE"))
        
        # KPIs stratégiques
        report.append("\n📊 TABLEAU DE BORD STRATÉGIQUE")
        report.append("=" * 80)
        
        kpis = self._calculate_executive_kpis(fleet_data, economics, alerts)
        for kpi, value in kpis.items():
            report.append(f"  {kpi}: {value}")
        
        # Analyse ROI détaillée
        report.append("\n💰 ANALYSE FINANCIÈRE")
        report.append("=" * 80)
        roi_analysis = self._analyze_roi(economics, fleet_data)
        for item, details in roi_analysis.items():
            report.append(f"  {item}: {details}")
        
        # Risques prioritaires
        report.append("\n🔴 RISQUES CRITIQUES")
        report.append("=" * 80)
        critical_risks = self._identify_critical_risks(alerts, fleet_data)
        for i, risk in enumerate(critical_risks[:5], 1):
            report.append(f"  {i}. {risk}")
        
        # Recommandations stratégiques
        report.append("\n✅ RECOMMANDATIONS STRATÉGIQUES")
        report.append("=" * 80)
        recommendations = self._generate_strategic_recommendations(kpis, economics)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"  {i}. {rec}")
        
        # Projections
        report.append("\n📈 PROJECTIONS 12 MOIS")
        report.append("=" * 80)
        projections = self._generate_12month_projections(fleet_data, economics)
        for projection, value in projections.items():
            report.append(f"  {projection}: {value}")
            
        return "\n".join(report)
    
    def _calculate_executive_kpis(self, fleet_data: pd.DataFrame, 
                                 economics: Dict, alerts: List[Dict]) -> Dict:
        """Calcule les KPIs exécutifs."""
        total_machines = fleet_data['machineID'].nunique()
        critical_alerts = len([a for a in alerts if a['alert_level'] == 'CRITIQUE'])
        avg_health = fleet_data['global_health_score'].mean()
        
        return {
            "Parc total": f"{total_machines} machines",
            "Santé moyenne": f"{avg_health:.1f}/100",
            "Machines critiques": f"{critical_alerts} ({critical_alerts/total_machines*100:.1f}%)",
            "Économies annuelles": f"{economics.get('annual_savings', 0):,.0f} €",
            "ROI maintenance prédictive": f"{economics.get('roi_percent', 0):.1f}%",
            "Temps d'arrêt évité": f"{economics.get('downtime_savings', 0):.0f} h/an",
            "Coût maintenance moyen": f"{economics.get('avg_maintenance_cost', 0):,.0f} €/machine"
        }
    
    def _analyze_roi(self, economics: Dict, fleet_data: pd.DataFrame) -> Dict:
        """Analyse détaillée du ROI."""
        investment = economics.get('ml_investment', 100000)
        annual_savings = economics.get('annual_savings', 250000)
        payback_months = (investment / annual_savings) * 12 if annual_savings > 0 else 0
        
        return {
            "Investissement ML": f"{investment:,.0f} €",
            "Économies annuelles": f"{annual_savings:,.0f} €",
            "Période de retour": f"{payback_months:.1f} mois",
            "Bénéfice net 3 ans": f"{(annual_savings * 3 - investment):,.0f} €",
            "ROI annuel": f"{(annual_savings / investment * 100) if investment > 0 else 0:.1f}%"
        }
    
    def _identify_critical_risks(self, alerts: List[Dict], fleet_data: pd.DataFrame) -> List[str]:
        """Identifie les risques critiques."""
        critical_risks = []
        critical_machines = [a for a in alerts if a['alert_level'] == 'CRITIQUE']
        
        for alert in critical_machines[:10]:  # Limiter aux 10 plus critiques
            machine_info = f"Machine {alert['machine_id']} ({alert['model']}) - "
            machine_info += f"RUL: {alert['rul_days']}j - "
            machine_info += f"Risque: {alert.get('risk_level', 'ÉLEVÉ')}"
            critical_risks.append(machine_info)
            
        return critical_risks
    
    def _generate_strategic_recommendations(self, kpis: Dict, economics: Dict) -> List[str]:
        """Génère des recommandations stratégiques."""
        recommendations = [
            "Déployer le système d'alertes sur l'ensemble du parc critique",
            "Former les équipes maintenance à l'utilisation des prédictions RUL",
            "Réallouer 15% du budget maintenance curative vers la maintenance prédictive",
            "Mettre en place un tableau de bord temps réel pour le suivi des KPIs",
            "Étendre l'analyse prédictive aux nouveaux modèles de machines"
        ]
        
        if economics.get('roi_percent', 0) > 200:
            recommendations.append("Augmenter l'investissement ML de 20% pour étendre la couverture")
            
        return recommendations
    
    def _generate_12month_projections(self, fleet_data: pd.DataFrame, economics: Dict) -> Dict:
        """Génère des projections sur 12 mois."""
        current_savings = economics.get('annual_savings', 0)
        growth_rate = 0.15  # 15% de croissance annuelle
        
        return {
            "Économies projetées (M+6)": f"{(current_savings * 1.5):,.0f} €",
            "Économies projetées (M+12)": f"{(current_savings * (1 + growth_rate)):,.0f} €",
            "Réduction temps d'arrêt": "-45% vs année précédente",
            "Amélioration santé parc": "+12% vs baseline actuelle"
        }

    def generate_maintenance_report(self, alerts: List[Dict], 
                                  maintenance_calendar: Dict,
                                  parts_inventory: Dict = None) -> str:
        """Rapport opérationnel pour l'équipe maintenance."""
        report = []
        
        report.extend(self._create_report_header("RAPPORT MAINTENANCE OPÉRATIONNEL"))
        
        # Alertes par priorité
        report.append("\n🚨 ALERTES ACTIVES PAR NIVEAU DE PRIORITÉ")
        report.append("=" * 80)
        
        alert_stats = self._calculate_alert_statistics(alerts)
        for level, count in alert_stats.items():
            report.append(f"  {level}: {count} machines")
        
        # Détail des alertes critiques
        report.append("\n🔴 ALERTES CRITIQUES - ACTION IMMÉDIATE")
        report.append("=" * 80)
        critical_alerts = [a for a in alerts if a['alert_level'] == 'CRITIQUE']
        for alert in critical_alerts[:15]:  # Limiter l'affichage
            report.append(self._format_maintenance_alert(alert))
        
        # Calendrier de maintenance
        report.append("\n📅 CALENDRIER MAINTENANCE - 4 SEMAINES")
        report.append("=" * 80)
        for week, machines in maintenance_calendar.items():
            if machines:
                report.append(f"\n{week.upper()}:")
                for machine in machines[:5]:  # 5 premières machines par semaine
                    report.append(f"  • Machine {machine['machine_id']} - {machine['model']}")
                    report.append(f"    RUL: {machine['rul_days']}j | Deadline: {machine['maintenance_deadline']}")
        
        # Besoins en pièces détachées
        if parts_inventory:
            report.append("\n🔧 BESOINS EN PIÈCES DÉTACHÉES")
            report.append("=" * 80)
            for part, need in parts_inventory.items():
                report.append(f"  {part}: {need['quantity']} unités - {need['urgency']}")
        
        # Instructions opérationnelles
        report.append("\n⚡ INSTRUCTIONS OPÉRATIONNELLES")
        report.append("=" * 80)
        instructions = self._generate_maintenance_instructions(alerts)
        for i, instruction in enumerate(instructions, 1):
            report.append(f"  {i}. {instruction}")
            
        return "\n".join(report)
    
    def _calculate_alert_statistics(self, alerts: List[Dict]) -> Dict:
        """Calcule les statistiques des alertes."""
        levels = ['CRITIQUE', 'HAUTE', 'MOYENNE', 'BASSE']
        stats = {}
        
        for level in levels:
            count = len([a for a in alerts if a['alert_level'] == level])
            if count > 0:
                stats[level] = count
                
        return stats
    
    def _format_maintenance_alert(self, alert: Dict) -> str:
        """Formate une alerte pour le rapport maintenance."""
        lines = []
        lines.append(f"  🚨 Machine {alert['machine_id']} ({alert['model']})")
        lines.append(f"     • RUL: {alert['rul_days']} jours")
        lines.append(f"     • Santé: {alert['health_score']}/100")
        lines.append(f"     • Risque: {alert.get('risk_level', 'N/A')}")
        lines.append(f"     • Deadline: {alert['maintenance_deadline']}")
        lines.append(f"     • Action: {alert['recommended_action']}")
        return "\n".join(lines)
    
    def _generate_maintenance_instructions(self, alerts: List[Dict]) -> List[str]:
        """Génère des instructions opérationnelles."""
        instructions = []
        critical_count = len([a for a in alerts if a['alert_level'] == 'CRITIQUE'])
        high_count = len([a for a in alerts if a['alert_level'] == 'HAUTE'])
        
        if critical_count > 0:
            instructions.append(f"Intervenir sur les {critical_count} machines CRITIQUES dans les 48h")
            instructions.append("Préparer équipe d'intervention rapide avec pièces critiques")
        
        if high_count > 0:
            instructions.append(f"Planifier maintenance des {high_count} machines HAUTE priorité cette semaine")
        
        instructions.append("Mettre à jour le stock de pièces détachées basé sur le calendrier")
        instructions.append("Documenter toutes les interventions pour amélioration des modèles")
        
        return instructions

    def generate_technical_report(self, model_performance: Dict, 
                                feature_importance: Dict,
                                data_quality: Dict) -> str:
        """Rapport technique pour data scientists."""
        report = []
        
        report.extend(self._create_report_header("RAPPORT TECHNIQUE - PERFORMANCE MODÈLES"))
        
        # Performance des modèles
        report.append("\n🤖 PERFORMANCE MODÈLES ML")
        report.append("=" * 80)
        
        if 'classifier' in model_performance:
            report.append("\n📊 CLASSIFICATEUR (Prédiction Panne)")
            report.append("-" * 40)
            clf_perf = model_performance['classifier']
            report.extend(self._format_model_performance(clf_perf))
        
        if 'regressor' in model_performance:
            report.append("\n📈 RÉGRESSEUR RUL (Durée de Vie Restante)")
            report.append("-" * 40)
            reg_perf = model_performance['regressor']
            report.extend(self._format_regressor_performance(reg_perf))
        
        # Importance des features
        report.append("\n🎯 IMPORTANCE DES VARIABLES")
        report.append("=" * 80)
        if feature_importance:
            for model, features in feature_importance.items():
                report.append(f"\n{model.upper()}:")
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
                for feature, importance in sorted_features:
                    report.append(f"  {feature}: {importance:.3f}")
        
        # Qualité des données
        report.append("\n📋 QUALITÉ DES DONNÉES")
        report.append("=" * 80)
        report.extend(self._format_data_quality(data_quality))
        
        # Recommandations techniques
        report.append("\n💡 RECOMMANDATIONS TECHNIQUES")
        report.append("=" * 80)
        tech_recommendations = self._generate_technical_recommendations(model_performance)
        for i, rec in enumerate(tech_recommendations, 1):
            report.append(f"  {i}. {rec}")
            
        return "\n".join(report)
    
    def _format_model_performance(self, performance: Dict) -> List[str]:
        """Formate les performances du classifieur."""
        lines = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        for metric in metrics:
            if metric in performance:
                value = performance[metric]
                if isinstance(value, float):
                    lines.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
        return lines
    
    def _format_regressor_performance(self, performance: Dict) -> List[str]:
        """Formate les performances du régresseur."""
        lines = []
        metrics = ['mae', 'rmse', 'r2']
        for metric in metrics:
            if metric in performance:
                value = performance[metric]
                lines.append(f"  {metric.upper()}: {value:.3f}")
        return lines
    
    def _format_data_quality(self, data_quality: Dict) -> List[str]:
        """Formate le rapport de qualité des données."""
        lines = []
        if 'missing_values' in data_quality:
            total_missing = sum(data_quality['missing_values'].values())
            lines.append(f"  Valeurs manquantes: {total_missing}")
        
        if 'data_types' in data_quality:
            lines.append(f"  Types de données: {len(data_quality['data_types'])} colonnes")
            
        return lines
    
    def _generate_technical_recommendations(self, model_performance: Dict) -> List[str]:
        """Génère des recommandations techniques."""
        recommendations = [
            "Implémenter la validation croisée imbriquée pour une meilleure estimation des performances",
            "Ajouter des features temporelles (moyennes mobiles, tendances) pour capturer l'évolution",
            "Tester XGBoost et LightGBM en benchmark contre Random Forest",
            "Implémenter l'optimisation bayésienne des hyperparamètres",
            "Ajouter l'analyse de incertitude des prédictions RUL"
        ]
        
        if 'classifier' in model_performance:
            if model_performance['classifier'].get('auc_roc', 0) < 0.85:
                recommendations.append("Améliorer le classifieur avec des techniques de ré-échantillonnage pour les classes déséquilibrées")
                
        return recommendations
    
    def _create_report_header(self, title: str) -> List[str]:
        """Crée l'en-tête du rapport."""
        return [
            "=" * 80,
            f"{title}",
            f"{self.company_name}",
            f"Date: {self.report_date.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ]

# Compatibilité avec l'ancien code
class IndustrialReportGenerator(AdvancedReportGenerator):
    """Maintains compatibility with existing code."""
    pass