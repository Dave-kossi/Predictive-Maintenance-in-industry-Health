"""src/llm_report.py - Version améliorée"""
import os
import json
import requests
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class AdvancedReportGenerator:
    """Générateur de rapports avancé avec templates et cache."""
    
    def __init__(self, hf_token: str = None, model: str = 'mistralai/Mistral-7B-Instruct'):
        self.hf_token = hf_token or os.environ.get('HF_API_TOKEN')
        self.model = model
        self.report_templates = self._load_templates()
        
    def _load_templates(self) -> Dict:
        """Charge les templates de rapports."""
        return {
            'executive': self._executive_template(),
            'technical': self._technical_template(),
            'maintenance': self._maintenance_template(),
            'alert': self._alert_template()
        }
    
    def _executive_template(self) -> str:
        """Template pour rapport direction."""
        return """Génère un rapport exécutif en français pour la direction.

CONTEXTE:
{context}

DONNÉES:
- Machines analysées: {total_machines}
- Machines critiques: {critical_machines}
- Santé moyenne: {avg_health}/100
- ROI estimé: {roi_percent}%

STRUCTURE DEMANDÉE:
1. Titre percutant
2. Résumé exécutif (3-4 lignes max)
3. Points clés (chiffrés)
4. Risques principaux
5. Recommandations stratégiques
6. Impact financier

Ton: Professionnel, concis, orienté décision."""

    def _technical_template(self) -> str:
        """Template pour rapport technique."""
        return """Génère un rapport technique détaillé en français.

ANALYSE EFFECTUÉE:
{analysis_details}

PERFORMANCE MODÈLES:
{model_performance}

STATISTIQUES:
{statistics}

STRUCTURE:
1. Méthodologie d'analyse
2. Performance des modèles ML
3. Insights techniques
4. Limitations
5. Améliorations recommandées

Public: Data Scientists et ingénieurs ML"""

    def build_advanced_prompt(self, data: Dict, report_type: str = 'executive') -> str:
        """Construit un prompt structuré selon le type de rapport."""
        template = self.report_templates.get(report_type, self.report_templates['executive'])
        
        # Enrichissement des données contextuelles
        enriched_data = self._enrich_report_data(data, report_type)
        
        return template.format(**enriched_data)
    
    def _enrich_report_data(self, data: Dict, report_type: str) -> Dict:
        """Enrichit les données pour le rapport."""
        base_context = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'report_type': report_type.upper(),
            'context': self._build_context_string(data)
        }
        
        # Données spécifiques au type de rapport
        if report_type == 'executive':
            base_context.update({
                'total_machines': data.get('total_machines', 'N/A'),
                'critical_machines': data.get('critical_count', 0),
                'avg_health': data.get('avg_health_score', 0),
                'roi_percent': data.get('roi_percent', 0)
            })
        elif report_type == 'technical':
            base_context.update({
                'analysis_details': self._format_technical_details(data),
                'model_performance': self._format_model_performance(data),
                'statistics': self._format_statistics(data)
            })
            
        return base_context
    
    def _build_context_string(self, data: Dict) -> str:
        """Construit le contexte à partir des données."""
        context_parts = []
        
        # Identification
        if 'identification' in data:
            ident = data['identification']
            context_parts.append(f"Machine: {ident.get('machine_id', 'N/A')} - Modèle: {ident.get('model', 'N/A')}")
        
        # Santé globale
        if 'sante_globale' in data:
            sante = data['sante_globale']
            context_parts.append(f"Score santé: {sante.get('score_global', 'N/A')}/100")
        
        # Prédictions risque
        if 'prediction_risque' in data:
            risque = data['prediction_risque']
            context_parts.append(f"Niveau risque: {risque.get('risk_level', 'N/A')} (confiance: {risque.get('confidence', 0):.1%})")
        
        # RUL
        if 'estimation_rul' in data:
            rul = data['estimation_rul']
            context_parts.append(f"RUL médian: {rul.get('rul_median', 'N/A')} jours")
        
        return " | ".join(context_parts)
    
    def _format_technical_details(self, data: Dict) -> str:
        """Formate les détails techniques."""
        details = []
        if 'model_performance' in data:
            perf = data['model_performance']
            details.append(f"Accuracy: {perf.get('accuracy', 0):.2%}")
            details.append(f"Precision: {perf.get('precision', 0):.2%}")
            details.append(f"Recall: {perf.get('recall', 0):.2%}")
        return "\n".join(details)
    
    def _format_model_performance(self, data: Dict) -> str:
        """Formate les performances des modèles."""
        if 'model_metrics' not in data:
            return "Aucune métrique disponible"
        
        metrics = data['model_metrics']
        lines = []
        for model, perf in metrics.items():
            lines.append(f"{model}: MAE={perf.get('mae', 0):.2f}, R²={perf.get('r2', 0):.2f}")
        return "\n".join(lines)
    
    def _format_statistics(self, data: Dict) -> str:
        """Formate les statistiques."""
        stats = data.get('statistics', {})
        return f"""
        Machines totales: {stats.get('total_machines', 0)}
        Pannes détectées: {stats.get('failures', 0)}
        Taux de défaillance: {stats.get('failure_rate', 0):.2%}
        Santé moyenne: {stats.get('avg_health', 0):.1f}/100
        """
    
    def generate_llm_report(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Génère un rapport via l'API Hugging Face avec gestion d'erreurs avancée."""
        if not self.hf_token:
            return self._generate_fallback_report(prompt)
        
        try:
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return self._extract_generated_text(result)
            else:
                error_msg = f"Erreur API: {response.status_code} - {response.text}"
                return self._generate_fallback_report(prompt, error_msg)
                
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            return self._generate_fallback_report(prompt, error_msg)
    
    def _extract_generated_text(self, api_result) -> str:
        """Extrait le texte généré de la réponse API."""
        if isinstance(api_result, list) and len(api_result) > 0:
            return api_result[0].get('generated_text', '')
        elif isinstance(api_result, dict):
            return api_result.get('generated_text', '')
        else:
            return str(api_result)
    
    def _generate_fallback_report(self, prompt: str, error: str = None) -> str:
        """Génère un rapport de fallback."""
        fallback_msg = "⚠️ RAPPORT AUTOMATIQUE (API non disponible)\n\n"
        if error:
            fallback_msg += f"Erreur: {error}\n\n"
        
        fallback_msg += "Prompt qui serait envoyé à l'IA:\n"
        fallback_msg += "=" * 50 + "\n"
        fallback_msg += prompt + "\n"
        fallback_msg += "=" * 50 + "\n\n"
        fallback_msg += "Pour générer le rapport complet, configurez HF_API_TOKEN"
        
        return fallback_msg
    
    def generate_comprehensive_report(self, data: Dict, report_type: str = 'executive', 
                                   out_path: str = None) -> str:
        """Génère un rapport complet."""
        prompt = self.build_advanced_prompt(data, report_type)
        report = self.generate_llm_report(prompt)
        
        if out_path:
            self._save_report(report, out_path)
            
        return report
    
    def _save_report(self, report: str, path: str):
        """Sauvegarde le rapport."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)

# Fonctions originales maintenues pour compatibilité
def build_prompt(summary: Dict) -> str:
    return AdvancedReportGenerator().build_advanced_prompt(summary, 'executive')

def generate_report_hf(prompt: str, model: str = 'mistralai/Mistral-7B-Instruct', 
                      max_tokens: int = 512) -> str:
    return AdvancedReportGenerator().generate_llm_report(prompt, max_tokens)

def generate_report(summary: Dict, out_path: str = None) -> str:
    generator = AdvancedReportGenerator()
    return generator.generate_comprehensive_report(summary, 'executive', out_path)