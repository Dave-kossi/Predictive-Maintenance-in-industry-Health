"""src/survival_analysis.py - Analyse de survie avancée"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from typing import Dict, List, Tuple, Optional
import warnings

from .utils import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')

class AdvancedSurvivalAnalyzer:
    """Analyse de survie avancée avec Kaplan-Meier et Cox."""
    
    def __init__(self):
        """Initialise l'analyseur."""
        self.kmf_dict = {}
        self.cox_fitter = None
        self.survival_results = {}
        self.logger = logger
        
    def fit_kmf_by_model(self, df: pd.DataFrame, time_col: str = 'time', 
                        event_col: str = 'event', group_col: str = 'model') -> Dict[str, KaplanMeierFitter]:
        """Entraîne Kaplan-Meier pour chaque groupe.
        
        Args:
            df: DataFrame avec colonnes time, event, et groupe
            time_col: Colonne temps de vie
            event_col: Colonne événement
            group_col: Colonne de groupement (défaut: 'model')
        
        Returns:
            Dict {groupe: KaplanMeierFitter}
        """
        self.kmf_dict = {}
        groups = df[group_col].unique()
        
        self.logger.info(f"Entraînement Kaplan-Meier pour {len(groups)} groupes...")
        
        for group in groups:
            try:
                kmf = KaplanMeierFitter()
                sub_df = df[df[group_col] == group]
                
                if len(sub_df) > 1:  # Au moins 2 observations
                    kmf.fit(
                        durations=sub_df[time_col],
                        event_observed=sub_df[event_col],
                        label=str(group)
                    )
                    self.kmf_dict[group] = kmf
                    
                    # Statistiques
                    events = int(sub_df[event_col].sum())
                    total = len(sub_df)
                    self.survival_results[group] = {
                        'median_survival_time': float(kmf.median_survival_time_) if kmf.median_survival_time_ else np.nan,
                        'survival_at_30': float(kmf.predict(30)) if 30 <= sub_df[time_col].max() else np.nan,
                        'survival_at_90': float(kmf.predict(90)) if 90 <= sub_df[time_col].max() else np.nan,
                        'survival_at_180': float(kmf.predict(180)) if 180 <= sub_df[time_col].max() else np.nan,
                        'total_events': events,
                        'total_observations': total,
                        'event_rate': float(events / total) if total > 0 else 0,
                    }
                    
                    self.logger.debug(f"  {group}: {events}/{total} événements")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur pour groupe {group}: {e}")
        
        self.logger.info(f"✅ Kaplan-Meier entraîné pour {len(self.kmf_dict)} groupes")
        return self.kmf_dict
    
    def compare_survival_curves(self, df: pd.DataFrame, group_col: str = 'model',
                               time_col: str = 'time', event_col: str = 'event') -> Dict:
        """Compare les courbes de survie entre groupes.
        
        Returns:
            Dict avec résultats de tests log-rank
        """
        groups = df[group_col].unique()
        comparison_results = {}
        
        self.logger.info(f"Comparaison des courbes ({len(groups)} groupes)...")
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                try:
                    data1 = df[df[group_col] == group1]
                    data2 = df[df[group_col] == group2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        result = logrank_test(
                            data1[time_col], data2[time_col],
                            event_observed_A=data1[event_col],
                            event_observed_B=data2[event_col]
                        )
                        
                        comparison_results[f"{group1}_vs_{group2}"] = {
                            'p_value': float(result.p_value),
                            'test_statistic': float(result.test_statistic),
                            'significant': result.p_value < 0.05,
                        }
                except Exception as e:
                    self.logger.warning(f"Erreur comparaison {group1} vs {group2}: {e}")
        
        self.logger.info(f"✅ {len(comparison_results)} comparaisons effectuées")
        return comparison_results
    
    def fit_cox(self, df: pd.DataFrame, duration_col: str = 'time', 
               event_col: str = 'event', feature_cols: List[str] = None) -> Dict:
        """Ajuste un modèle de risques proportionnels de Cox.
        
        Args:
            df: DataFrame
            duration_col: Colonne durée
            event_col: Colonne événement
            feature_cols: Colonnes features
        
        Returns:
            Résumé du modèle
        """
        try:
            self.cox_fitter = CoxPHFitter()
            
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c not in [duration_col, event_col, 'model', 'machineID']]
            
            df_cox = df[[duration_col, event_col] + feature_cols].dropna()
            
            if len(df_cox) < 3:
                self.logger.warning("Pas assez de données pour Cox")
                return {}
            
            self.cox_fitter.fit(
                df_cox,
                duration_col=duration_col,
                event_col=event_col
            )
            
            self.logger.info("✅ Modèle Cox ajusté")
            return self.cox_fitter.summary.to_dict()
            
        except Exception as e:
            self.logger.warning(f"⚠️  Erreur ajustement Cox: {e}")
            return {}
    
    def create_interactive_plot(self, figsize: Tuple[int, int] = (12, 6),
                               save_path: str = None) -> go.Figure:
        """Crée un graphique interactif Plotly des courbes Kaplan-Meier.
        
        Args:
            figsize: Taille du graphique (ignorée pour Plotly)
            save_path: Chemin pour sauvegarder le graphique (HTML)
        
        Returns:
            Figure Plotly
        """
        if not self.kmf_dict:
            self.logger.warning("Aucune courbe KM disponible")
            return None
        
        fig = go.Figure()
        
        for group, kmf in self.kmf_dict.items():
            try:
                x = kmf.timeline.values
                y = kmf.survival_function_[str(group)].values
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    name=f'Modèle {group}',
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Temps: %{x:.0f} jours<br>' +
                                'Survie: %{y:.1%}<extra></extra>'
                ))
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur graphique pour {group}: {e}")
        
        fig.update_layout(
            title='Courbes de Survie Kaplan-Meier par Modèle',
            xaxis_title='Temps (jours)',
            yaxis_title='Probabilité de Survie',
            hovermode='x unified',
            template='plotly_white',
            height=600,
        )
        
        if save_path:
            try:
                fig.write_html(save_path)
                self.logger.info(f"✅ Graphique sauvegardé: {save_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur sauvegarde graphique: {e}")
        
        return fig
    
    def get_summary(self) -> Dict:
        """Retourne un résumé des résultats."""
        return {
            'n_groups': len(self.kmf_dict),
            'survival_results': self.survival_results,
            'cox_summary': self.cox_fitter.summary.to_dict() if self.cox_fitter else None,
        }


# ==================== FONCTIONS PUBLIQUES ====================

def fit_kmf_by_model(df: pd.DataFrame, time_col: str = 'time', 
                     event_col: str = 'event') -> Dict[str, KaplanMeierFitter]:
    """Charge et entraîne Kaplan-Meier par modèle de machine."""
    return AdvancedSurvivalAnalyzer().fit_kmf_by_model(df, time_col, event_col)

def compare_models(df: pd.DataFrame) -> Dict:
    """Compare les courbes de survie entre modèles."""
    return AdvancedSurvivalAnalyzer().compare_survival_curves(df)

logger.info("✅ Module survival_analysis chargé avec succès")

def plot_kmf(kmf_dict: Dict[str, KaplanMeierFitter], save_path: str = None):
    """Version améliorée avec options de personnalisation."""
    plt.figure(figsize=(10, 6))
    
    for model, kmf in kmf_dict.items():
        kmf.plot_survival_function(ci_show=True, label=f'Modèle {model}')
    
    plt.title('Analyse de Survie Kaplan-Meier par Modèle', fontsize=14, fontweight='bold')
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Probabilité de Survie', fontsize=12)
    plt.legend(title='Modèles')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()