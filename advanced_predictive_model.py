# advanced_predictive_model.py
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictiveMaintenance:
    """Algorithme avancé de maintenance prédictive - VERSION STABLE"""
    
    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.scaler = StandardScaler()
        self.data = None
        self.is_trained = False
        
    def load_data(self, file_path="data/Predictive_Table.csv"):
        """Charge et prépare les données"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"✅ Données chargées: {len(self.data)} machines")
            self._create_advanced_features()
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def _create_advanced_features(self):
        """Crée des indicateurs avancés - VERSION SIMPLIFIÉE"""
        
        # Scores de santé simples
        self.data['vibration_score'] = np.maximum(0, 100 - (self.data['vibration'] - 35) * 2)
        self.data['pressure_score'] = np.maximum(0, 100 - abs(self.data['pressure'] - 100) * 2)
        self.data['error_score'] = np.maximum(0, 100 - self.data['error_count'] * 2)
        self.data['maintenance_score'] = np.minimum(self.data['maint_count'] * 3, 100)
        self.data['age_score'] = np.maximum(0, 100 - self.data['age'] * 5)
        
        # Score de santé global
        self.data['global_health_score'] = (
            self.data['vibration_score'] * 0.3 +
            self.data['pressure_score'] * 0.25 +
            self.data['error_score'] * 0.2 +
            self.data['maintenance_score'] * 0.15 +
            self.data['age_score'] * 0.1
        ).clip(0, 100)
        
        # Indicateur de dégradation
        self.data['degradation_rate'] = self.data['vibration'] / (self.data['age'] + 1)
        
        print("✅ Features créées")
    
    def _create_risk_labels(self):
        """Crée les labels de risque - VERSION CORRIGÉE"""
        health_scores = self.data['global_health_score'].values
        vibrations = self.data['vibration'].values
        error_counts = self.data['error_count'].values
        
        risk_labels = []
        
        for i in range(len(self.data)):
            health = health_scores[i]
            vib = vibrations[i]
            errors = error_counts[i]
            
            if health < 30 or vib > 45 or errors > 40:
                risk_labels.append('CRITIQUE')
            elif health < 50 or vib > 40 or errors > 30:
                risk_labels.append('ÉLEVÉ')
            elif health < 70:
                risk_labels.append('MODÉRÉ')
            else:
                risk_labels.append('FAIBLE')
        
        return risk_labels
    
    def train_models(self):
        """Entraîne les modèles - VERSION STABLE"""
        print("🤖 Entraînement des modèles...")
        
        try:
            # 1. Kaplan-Meier
            self.kmf.fit(durations=self.data['time'], event_observed=self.data['event'])
            print("✅ Kaplan-Meier entraîné")
            
            # 2. Classification des risques
            X_class = self.data[['global_health_score', 'vibration', 'error_count', 'degradation_rate']]
            y_class = self._create_risk_labels()
            
            self.rf_classifier.fit(X_class, y_class)
            print("✅ Classificateur de risque entraîné")
            
            # 3. Clustering simple
            X_cluster = self.data[['vibration', 'pressure', 'error_count']].fillna(0)
            X_scaled = self.scaler.fit_transform(X_cluster)
            
            # Réduire le nombre de clusters si peu de données
            n_clusters = min(3, len(self.data) // 10)
            if n_clusters < 2:
                n_clusters = 2
                
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.data['failure_pattern'] = self.kmeans.fit_predict(X_scaled)
            print("✅ Clustering terminé")
            
            self.is_trained = True
            print("🎯 Tous les modèles entraînés avec succès")
            
        except Exception as e:
            print(f"❌ Erreur entraînement: {e}")
            # Mode dégradé avec Kaplan-Meier seulement
            self.kmf.fit(durations=self.data['time'], event_observed=self.data['event'])
            self.is_trained = True
    
    def predict_machine_risk(self, machine_id):
        """Prédit le niveau de risque"""
        if not self.is_trained:
            return {"risk_level": "INCONNU", "confidence": 0.0}
        
        try:
            machine_data = self.data[self.data['machineID'] == machine_id].iloc[0]
            
            X_pred = [[
                machine_data['global_health_score'],
                machine_data['vibration'],
                machine_data['error_count'],
                machine_data['degradation_rate']
            ]]
            
            risk_level = self.rf_classifier.predict(X_pred)[0]
            risk_proba = self.rf_classifier.predict_proba(X_pred)[0]
            
            return {
                'risk_level': risk_level,
                'confidence': float(max(risk_proba)),
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(self.rf_classifier.classes_, risk_proba)
                }
            }
        except:
            # Fallback basé sur le score de santé
            machine_data = self.data[self.data['machineID'] == machine_id].iloc[0]
            health = machine_data['global_health_score']
            
            if health < 30:
                risk_level = 'CRITIQUE'
            elif health < 50:
                risk_level = 'ÉLEVÉ'
            elif health < 70:
                risk_level = 'MODÉRÉ'
            else:
                risk_level = 'FAIBLE'
                
            return {'risk_level': risk_level, 'confidence': 0.8}
    
    def estimate_rul(self, machine_id):
        """Estime le temps restant avant panne"""
        if not self.is_trained:
            return {"rul_median": 100, "survival_30d": 0.8, "survival_90d": 0.6, "current_survival_time": 0}
        
        try:
            machine_data = self.data[self.data['machineID'] == machine_id].iloc[0]
            
            # Estimation simple basée sur Kaplan-Meier
            median_rul = self.kmf.median_survival_time_
            survival_30d = float(self.kmf.predict(30))
            survival_90d = float(self.kmf.predict(90))
            
            return {
                'rul_median': round(float(median_rul), 1),
                'survival_30d': round(survival_30d, 3),
                'survival_90d': round(survival_90d, 3),
                'current_survival_time': float(machine_data['time'])
            }
        except:
            return {"rul_median": 100, "survival_30d": 0.8, "survival_90d": 0.6, "current_survival_time": 0}
    
    def detect_anomalies(self, machine_id):
        """Détecte les événements anormaux"""
        try:
            machine_data = self.data[self.data['machineID'] == machine_id].iloc[0]
            anomalies = []
            
            if machine_data['vibration'] > 45:
                anomalies.append({
                    'type': 'VIBRATION_ÉLEVÉE',
                    'severity': 'HAUTE',
                    'message': f"Vibration élevée: {machine_data['vibration']:.1f}",
                    'recommendation': 'Vérifier équilibrage'
                })
            
            if machine_data['pressure'] < 95 or machine_data['pressure'] > 105:
                anomalies.append({
                    'type': 'PRESSION_ANORMALE',
                    'severity': 'MOYENNE',
                    'message': f"Pression anormale: {machine_data['pressure']:.1f}",
                    'recommendation': 'Contrôler régulateur'
                })
            
            if machine_data['error_count'] > 40:
                anomalies.append({
                    'type': 'ERREURS_EXCESSIVES',
                    'severity': 'HAUTE',
                    'message': f"Erreurs excessives: {machine_data['error_count']}",
                    'recommendation': 'Analyser logs'
                })
            
            return anomalies
        except:
            return []
    
    def generate_maintenance_recommendations(self, machine_id):
        """Génère des recommandations"""
        try:
            risk_prediction = self.predict_machine_risk(machine_id)
            rul_estimation = self.estimate_rul(machine_id)
            anomalies = self.detect_anomalies(machine_id)
            
            recommendations = []
            risk_level = risk_prediction['risk_level']
            
            if risk_level == 'CRITIQUE':
                recommendations.extend([
                    "Maintenance immédiate requise",
                    "Inspection complète des composants",
                    "Arrêt recommandé si possible"
                ])
            elif risk_level == 'ÉLEVÉ':
                recommendations.extend([
                    "Maintenance sous 7 jours",
                    "Surveillance renforcée",
                    "Analyse des causes"
                ])
            elif risk_level == 'MODÉRÉ':
                recommendations.extend([
                    "Maintenance dans 30 jours",
                    "Monitoring des tendances",
                    "Contrôle des paramètres"
                ])
            else:
                recommendations.extend([
                    "Maintenance à 60 jours",
                    "Surveillance normale",
                    "Opérations standards"
                ])
            
            for anomaly in anomalies:
                recommendations.append(anomaly['recommendation'])
            
            if rul_estimation['rul_median'] < 30:
                recommendations.append("RUL faible - Intervention urgente")
            
            return recommendations
        except:
            return ["Maintenance standard recommandée"]
    
    def get_machine_comprehensive_analysis(self, machine_id):
        """Analyse complète d'une machine"""
        try:
            machine_data = self.data[self.data['machineID'] == machine_id].iloc[0]
            
            return {
                'identification': {
                    'machine_id': int(machine_id),
                    'model': str(machine_data['model']),
                    'age': int(machine_data['age']),
                    'pattern_failure': int(machine_data.get('failure_pattern', 0))
                },
                'sante_globale': {
                    'score_global': float(round(machine_data['global_health_score'], 1)),
                    'sante_mecanique': float(round(machine_data['vibration_score'], 1)),
                    'sante_pression': float(round(machine_data['pressure_score'], 1))
                },
                'prediction_risque': self.predict_machine_risk(machine_id),
                'estimation_rul': self.estimate_rul(machine_id),
                'anomalies_detectees': self.detect_anomalies(machine_id),
                'recommandations_maintenance': self.generate_maintenance_recommendations(machine_id),
                'indicateurs_techniques': {
                    'vibration': float(round(machine_data['vibration'], 1)),
                    'pression': float(round(machine_data['pressure'], 1)),
                    'rotation': float(round(machine_data['rotate'], 1)),
                    'erreurs_cumulees': int(machine_data['error_count']),
                    'maintenances_realisees': int(machine_data['maint_count'])
                }
            }
        except Exception as e:
            print(f"❌ Erreur analyse machine {machine_id}: {e}")
            return {
                'identification': {'machine_id': machine_id, 'model': 'INCONNU', 'age': 0, 'pattern_failure': 0},
                'sante_globale': {'score_global': 50, 'sante_mecanique': 50, 'sante_pression': 50},
                'prediction_risque': {'risk_level': 'MODÉRÉ', 'confidence': 0.5},
                'estimation_rul': {'rul_median': 100, 'survival_30d': 0.8, 'survival_90d': 0.6},
                'anomalies_detectees': [],
                'recommandations_maintenance': ["Analyse indisponible"],
                'indicateurs_techniques': {'vibration': 0, 'pression': 0, 'rotation': 0, 'erreurs_cumulees': 0, 'maintenances_realisees': 0}
            }
    
    def get_fleet_risk_distribution(self):
        """Distribution des risques dans le parc"""
        try:
            risk_levels = []
            for machine_id in self.data['machineID']:
                pred = self.predict_machine_risk(machine_id)
                risk_levels.append(pred['risk_level'])
            
            risk_counts = pd.Series(risk_levels).value_counts()
            
            critical_machines = []
            for i, machine_id in enumerate(self.data['machineID']):
                if risk_levels[i] in ['CRITIQUE', 'ÉLEVÉ']:
                    critical_machines.append(int(machine_id))
            
            return {
                'distribution': risk_counts.to_dict(),
                'machines_critiques': critical_machines
            }
        except:
            return {'distribution': {'MODÉRÉ': len(self.data)}, 'machines_critiques': []}