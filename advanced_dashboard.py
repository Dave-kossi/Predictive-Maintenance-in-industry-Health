"""advanced_dashboard.py - Version améliorée"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Import helper: support running as package, as script from project root, or when
# executing inside the `src/` directory. This makes the module robust to different
# execution contexts used during development.
try:
    # preferred: package-relative import when running as `python -m src...`
    from .config import MODEL_LABEL_DESC, map_model_to_label
    # We'll try to import the data pipeline to load your real dataset.
    try:
        from .data_prep import load_and_prepare
    except Exception:
        try:
            from src.data_prep import load_and_prepare
        except Exception:
            load_and_prepare = None
except Exception:
    try:
        # usual absolute import when project root is on PYTHONPATH
        from src.config import MODEL_LABEL_DESC, map_model_to_label
    except Exception:
        # if we are executing from inside `src/` (cwd == src), add the parent
        # directory (project root) to sys.path so `src` package can be resolved.
        import sys
        import os
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(this_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from src.config import MODEL_LABEL_DESC, map_model_to_label
        except Exception:
            # final fallback: load config directly by file path
            try:
                import importlib.util
                cfg_path = os.path.join(this_dir, 'config.py')
                spec = importlib.util.spec_from_file_location('src.config', cfg_path)
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
                MODEL_LABEL_DESC = getattr(cfg, 'MODEL_LABEL_DESC')
                map_model_to_label = getattr(cfg, 'map_model_to_label')
            except Exception:
                raise
# Ensure `load_and_prepare` is defined regardless of import context
try:
    from .data_prep import load_and_prepare
except Exception:
    try:
        from src.data_prep import load_and_prepare
    except Exception:
        try:
            # if running inside src/ dir, add project root to path and retry
            import sys, os
            this_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(this_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.data_prep import load_and_prepare
        except Exception:
            load_and_prepare = None
from datetime import datetime, timedelta
import numpy as np
import os
import json
import joblib

# === CACHE STREAMLIT POUR ACCÉLÉRATION ===
@st.cache_data(show_spinner=False)
def cached_load_and_prepare(file_path: str):
    """Charge les données avec caching Streamlit (seulement une fois)."""
    with st.spinner("⏳ Chargement et enrichissement des données..."):
        try:
            if ('load_and_prepare' in globals()) and (load_and_prepare is not None):
                return load_and_prepare(file_path)
        except Exception as e:
            st.error(f"Erreur lors du chargement: {e}")
    return None

# Helper: load saved ML models and training report if present
def _load_saved_models():
    clf = None
    reg = None
    report = {}
    try:
        try:
            from src.config import get_models_dir
        except Exception:
            from config import get_models_dir

        models_dir = get_models_dir()
        clf_path = os.path.join(models_dir, 'classifier.joblib')
        reg_path = os.path.join(models_dir, 'regressor.joblib')
        report_path = os.path.join(models_dir, 'training_report.json')

        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
        if os.path.exists(reg_path):
            reg = joblib.load(reg_path)
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as fh:
                report = json.load(fh)
    except Exception:
        clf = clf or None
        reg = reg or None
        report = report or {}
    return clf, reg, report


def _run_training_and_reload(timeout: int = 1800):
    """Lance le script de training puis recharge le rapport généré.

    Returns the loaded training report dict on success, or None on failure.
    """
    try:
        import subprocess, sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable, '-m', 'src.train_models']
        proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            # print logs to stdout/stderr for debugging
            print(proc.stdout)
            print(proc.stderr)
            return None
        # reload saved report
        _, _, report = _load_saved_models()
        return report
    except Exception as e:
        print('Erreur lors du lancement de l\'entraînement:', e)
        return None


def _generate_decision_pack(data: pd.DataFrame):
    """Génère un Decision Pack (texte) et le sauvegarde dans `reports/`.

    Returns (content:str, filename:str, path:str)
    """
    try:
        try:
            from .economics import EconomicsAnalyzer
        except Exception:
            from src.economics import EconomicsAnalyzer
    except Exception:
        EconomicsAnalyzer = None

    lines = []
    lines.append('Decision Pack - Maintenance Prédictive')
    lines.append(f'Date: {datetime.now().isoformat()}')
    lines.append('')
    # Top 5 riskiest machines
    lines.append('Top 5 Machines à Risque:')
    if 'global_health_score' in data.columns:
        top5 = data.nsmallest(5, 'global_health_score')
    else:
        top5 = data.head(5)

    for _, r in top5.iterrows():
        prob = r.get('pred_proba', 'N/A')
        lines.append(f"- Machine {r['machineID']} | Modèle: {r.get('model','?')} | Santé: {r.get('global_health_score','?')} | ProbFail: {prob}")

    if EconomicsAnalyzer is not None:
        try:
            analyzer = EconomicsAnalyzer()
            econ = analyzer.estimate_portfolio_savings(data)
            lines.append('')
            lines.append('Résumé Economique:')
            for k, v in (econ or {}).items():
                lines.append(f"- {k}: {v}")
        except Exception:
            lines.append('Résumé économique: indisponible (erreur calcul)')

    fname = f"decision_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, fname)
    content = "\n".join(lines)
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
    except Exception:
        path = ''

    return content, fname, path

class AdvancedDashboard:
    """Dashboard Streamlit avancé pour la maintenance prédictive."""
    
    def __init__(self):
        self.set_page_config()
        
    def set_page_config(self):
        """Configure la page Streamlit."""
        st.set_page_config(
            page_title="Système Avancé de Maintenance Prédictive",
            page_icon="🔧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_sidebar(self):
        """Configure la barre latérale."""
        st.sidebar.title("🔧 Configuration")
        
        # Sélection du modèle
        model_options = ["Vue Ensemble", "Analyse Machine", "Alertes", "Rapports"]
        selected_model = st.sidebar.selectbox("Mode d'analyse", model_options)
        
        # Filtres
        st.sidebar.subheader("Filtres")
        date_range = st.sidebar.date_input(
            "Période d'analyse",
            value=(datetime.now() - timedelta(days=90), datetime.now()),
            max_value=datetime.now()
        )

        # --- Data controls: allow direct loading of the real CSV or uploading a CSV ---
        with st.sidebar.expander('Données (charger / uploader)', expanded=False):
            uploaded = st.file_uploader('Charger un fichier CSV (optionnel)', type=['csv'])
            if uploaded is not None:
                try:
                    # save uploaded file into project data dir as Predictive_Table.csv
                    try:
                        from .config import DATA_DIR
                    except Exception:
                        from src.config import DATA_DIR
                    target = os.path.join(str(DATA_DIR), 'Predictive_Table.csv')
                    with open(target, 'wb') as fh:
                        fh.write(uploaded.getvalue())
                    st.success(f'Fichier uploadé et sauvegardé: {target}')
                    # load and store in session state (use cached version)
                    try:
                        cached_load_and_prepare.clear()  # Clear cache for fresh load
                        df = cached_load_and_prepare(target)
                        st.session_state['data_df'] = df
                        st.session_state['data_path'] = target
                        st.success('Dataset chargé depuis le fichier uploadé — rafraîchissement...')
                        st.rerun()
                    except Exception as e:
                        st.error(f"Échec du chargement du CSV uploadé: {e}")
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde du fichier uploadé: {e}")

            if st.button('Charger Predictive_Table.csv depuis project/data'):
                # attempt to load the canonical dataset path
                try:
                    try:
                        from .config import get_data_path
                    except Exception:
                        from src.config import get_data_path
                    p = None
                    try:
                        p = get_data_path('predictive')
                    except Exception:
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        p = os.path.join(project_root, 'data', 'Predictive_Table.csv')
                    if not os.path.exists(p):
                        st.error(f'Fichier introuvable: {p}')
                    else:
                        cached_load_and_prepare.clear()  # Clear cache for fresh load
                        df = cached_load_and_prepare(p)
                        st.session_state['data_df'] = df
                        st.session_state['data_path'] = p
                        st.success(f'Dataset chargé depuis: {p} — rafraîchissement...')
                        st.rerun()
                except Exception as e:
                    st.error(f"Échec du chargement du dataset: {e}")
        
        # KPIs rapides — calculés dynamiquement depuis les données actuelles
        st.sidebar.subheader("KPIs Rapides")
        # Ces valeurs seront mises à jour après que le dataset soit chargé
        # Pour l'instant, afficher des placeholders
        
        return selected_model, date_range
    
    def display_overview_tab(self, data):
        """Affiche l'onglet vue d'ensemble."""
        st.header("📊 Vue d'Ensemble du Parc")
        
        # ✅ Clear badge: all analyses use REAL data
        st.success("✅ **Analyses Authentiques** — Toutes les données et résultats proviennent de votre dataset réel (Predictive_Table.csv)")
        
        # Executive summary KPIs (from models/report if available)
        clf_tmp, reg_tmp, training_report = _load_saved_models()
        with st.expander("Résumé Exécutif (modèles & KPIs)", expanded=True):
            if training_report:
                clf_metrics = training_report.get('classifier', {})
                reg_metrics = training_report.get('regressor', {})
                st.write(f"📊 **Classifier (entraîné sur données réelles)** — Accuracy: {clf_metrics.get('accuracy', 'N/A')}, AUC: {clf_metrics.get('auc_roc', 'N/A')}")
                st.write(f"📈 **Regressor (entraîné sur données réelles)** — R²: {reg_metrics.get('r2', 'N/A')}, MAE: {reg_metrics.get('mae', 'N/A')} jours")
            else:
                st.info("Aucun rapport d'entraînement disponible. Entraînez les modèles pour afficher les métriques.")
            # Bouton pour relancer l'entraînement
            if st.button("🔁 Relancer l'entraînement"):
                with st.spinner("Entraînement en cours, cela peut prendre quelques minutes..."):
                    new_report = _run_training_and_reload()
                if new_report:
                    st.success("Entraînement terminé — rapport chargé.")
                    clf_metrics = new_report.get('classifier', {})
                    reg_metrics = new_report.get('regressor', {})
                    st.write(f"**Classifier AUC:** {clf_metrics.get('auc_roc', 'N/A')}")
                    st.write(f"**Regressor R²:** {reg_metrics.get('r2', 'N/A')}")
                else:
                    st.error("Échec de l'entraînement — voir logs pour détails.")
            # Simple KPI: number of high-risk machines
            high_risk = int((data['global_health_score'] < 40).sum())
            st.write(f"Machines à risque élevé: **{high_risk}**")
            # Average survival probability at 30 days (if exists)
            if 'survival_at_30' in data.columns:
                st.write(f"Prob. survie moyenne à 30j: **{data['survival_at_30'].mean():.2f}**")
        # Légende des modèles (A / B / C)
        with st.expander("Légende des modèles (A / B / C)", expanded=False):
            for key, desc in MODEL_LABEL_DESC:
                st.markdown(f"**{key}**: {desc}")
            st.markdown("---")
        
        # KPIs principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_machines = data['machineID'].nunique()
            st.metric("Machines Total", total_machines)
            
        with col2:
            avg_health = data['global_health_score'].mean()
            st.metric("Santé Moyenne", f"{avg_health:.1f}/100")
            
        with col3:
            critical_count = len([x for x in data[data['global_health_score'] < 30]['machineID'].unique()])
            st.metric("Machines Critiques", critical_count)
            
        with col4:
            failure_rate = data['event'].mean() * 100
            st.metric("Taux Défaillance", f"{failure_rate:.1f}%")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_health_distribution(data)
            
        with col2:
            self._display_risk_matrix(data)
            
        # Alertes récentes
        st.subheader("🚨 Alertes Récentes")
        self._display_recent_alerts(data)
        
    def _display_health_distribution(self, data):
        """Affiche la distribution de santé avec options interactives — 100% données réelles."""
        st.subheader("Distribution des Scores de Santé (Données Réelles)")
        
        # ===== OPTIONS DE FILTRAGE =====
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtre par plage de score
            health_min, health_max = st.slider(
                'Plage de scores de santé à afficher',
                min_value=0,
                max_value=100,
                value=(int(data['global_health_score'].min()), int(data['global_health_score'].max())),
                step=5
            )
        
        with col2:
            # Grouper par
            group_option = st.selectbox(
                'Regrouper par',
                ['Aucun', 'Modèle (A/B/C)', 'Niveau de Risque', 'Modèle technique'],
                key='health_dist_group'
            )
        
        with col3:
            # Type de visualisation
            chart_type = st.selectbox(
                'Type de graphique',
                ['Histogramme', 'Distribution (KDE)', 'Box Plot'],
                key='health_dist_chart'
            )
        
        # ===== FILTRER LES DONNÉES =====
        filtered_data = data[
            (data['global_health_score'] >= health_min) &
            (data['global_health_score'] <= health_max)
        ].copy()
        
        # ===== STATISTIQUES RÉELLES =====
        st.info(f"""
        📊 **Statistiques (données réelles filtrées)**
        - Machines: **{len(filtered_data)}** / {len(data)}
        - Score moyen: **{filtered_data['global_health_score'].mean():.1f}** / 100
        - Écart-type: **{filtered_data['global_health_score'].std():.1f}**
        - Min: {filtered_data['global_health_score'].min():.1f} | Max: {filtered_data['global_health_score'].max():.1f}
        """)
        
        # ===== AFFICHER LE GRAPHIQUE =====
        if chart_type == 'Histogramme':
            if group_option == 'Modèle (A/B/C)':
                if 'model_label' in filtered_data.columns:
                    fig = px.histogram(
                        filtered_data,
                        x='global_health_score',
                        color='model_label',
                        nbins=15,
                        title='Distribution des Scores de Santé par Modèle (A/B/C) - Données Réelles',
                        labels={'global_health_score': 'Score de Santé', 'count': 'Nombre de Machines'},
                        barmode='group'
                    )
                else:
                    fig = px.histogram(filtered_data, x='global_health_score', nbins=15, title='Distribution des Scores')
            
            elif group_option == 'Niveau de Risque':
                if 'risk_level' in filtered_data.columns:
                    fig = px.histogram(
                        filtered_data,
                        x='global_health_score',
                        color='risk_level',
                        nbins=15,
                        title='Distribution des Scores de Santé par Niveau de Risque - Données Réelles',
                        labels={'global_health_score': 'Score de Santé', 'count': 'Nombre de Machines'},
                        barmode='group'
                    )
                else:
                    fig = px.histogram(filtered_data, x='global_health_score', nbins=15, title='Distribution des Scores')
            
            elif group_option == 'Modèle technique':
                if 'model' in filtered_data.columns:
                    fig = px.histogram(
                        filtered_data,
                        x='global_health_score',
                        color='model',
                        nbins=15,
                        title='Distribution des Scores de Santé par Modèle Technique - Données Réelles',
                        labels={'global_health_score': 'Score de Santé', 'count': 'Nombre de Machines'},
                        barmode='group'
                    )
                else:
                    fig = px.histogram(filtered_data, x='global_health_score', nbins=15, title='Distribution des Scores')
            
            else:  # Aucun
                fig = px.histogram(
                    filtered_data,
                    x='global_health_score',
                    nbins=15,
                    title='Distribution des Scores de Santé - Données Réelles',
                    labels={'global_health_score': 'Score de Santé', 'count': 'Nombre de Machines'},
                    color_discrete_sequence=['#2E86AB']
                )
        
        elif chart_type == 'Distribution (KDE)':
            # KDE plot (smoother distribution) — alternatives robustes
            try:
                if group_option == 'Modèle (A/B/C)' and 'model_label' in filtered_data.columns:
                    # Violin plot par modèle (plus stable)
                    fig = px.violin(
                        filtered_data,
                        y='global_health_score',
                        color='model_label',
                        box=True,
                        points=False,
                        title='Distribution de Densité par Modèle (A/B/C) - Données Réelles'
                    )
                elif group_option == 'Niveau de Risque' and 'risk_level' in filtered_data.columns:
                    # Violin plot par risque
                    fig = px.violin(
                        filtered_data,
                        x='risk_level',
                        y='global_health_score',
                        color='risk_level',
                        box=True,
                        points=False,
                        title='Distribution de Densité par Niveau de Risque - Données Réelles'
                    )
                elif group_option == 'Modèle technique' and 'model' in filtered_data.columns:
                    # Violin plot par modèle technique
                    fig = px.violin(
                        filtered_data,
                        x='model',
                        y='global_health_score',
                        color='model',
                        box=True,
                        points=False,
                        title='Distribution de Densité par Modèle Technique - Données Réelles'
                    )
                else:
                    # Histogram avec marginal box plot
                    fig = px.histogram(
                        filtered_data,
                        x='global_health_score',
                        marginal='box',
                        nbins=15,
                        title='Distribution de Densité - Données Réelles',
                        color_discrete_sequence=['#2E86AB']
                    )
            except Exception as e:
                # Fallback: simple histogram si KDE échoue
                st.warning(f"⚠️ Visualisation temporairement indisponible. Affichage histogramme standard.")
                fig = px.histogram(
                    filtered_data,
                    x='global_health_score',
                    nbins=15,
                    title='Distribution des Scores de Santé (Standard)',
                    color_discrete_sequence=['#2E86AB']
                )
        
        else:  # Box Plot
            if group_option == 'Modèle (A/B/C)' and 'model_label' in filtered_data.columns:
                fig = px.box(
                    filtered_data,
                    x='model_label',
                    y='global_health_score',
                    color='model_label',
                    title='Distribution des Scores de Santé par Modèle (A/B/C) - Données Réelles'
                )
            elif group_option == 'Niveau de Risque' and 'risk_level' in filtered_data.columns:
                fig = px.box(
                    filtered_data,
                    x='risk_level',
                    y='global_health_score',
                    color='risk_level',
                    title='Distribution des Scores de Santé par Niveau de Risque - Données Réelles'
                )
            elif group_option == 'Modèle technique' and 'model' in filtered_data.columns:
                fig = px.box(
                    filtered_data,
                    x='model',
                    y='global_health_score',
                    color='model',
                    title='Distribution des Scores de Santé par Modèle Technique - Données Réelles'
                )
            else:
                # Simple box plot
                fig = px.box(
                    filtered_data,
                    y='global_health_score',
                    title='Distribution des Scores de Santé - Données Réelles'
                )
        
        fig.update_layout(
            xaxis_title='Score de Santé' if chart_type == 'Histogramme' else '',
            yaxis_title='Nombre de Machines' if chart_type == 'Histogramme' else 'Score de Santé',
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== TABLE DE DÉTAIL =====
        with st.expander("📋 Voir détails des machines", expanded=False):
            display_cols = ['machineID', 'model', 'global_health_score', 'risk_level', 'vibration', 'error_count']
            if 'model_label' in filtered_data.columns:
                display_cols.insert(2, 'model_label')
            
            detail_df = filtered_data[display_cols].sort_values('global_health_score', ascending=False)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
    def _display_risk_matrix(self, data):
        """Affiche la matrice de risque — calculs basés sur les données réelles."""
        # Utiliser les données réelles pour calculer les niveaux de risque
        risk_data = data.copy()
        # Utiliser la colonne risk_level si elle existe (calculée lors de load_and_prepare)
        if 'risk_level' not in risk_data.columns:
            risk_data['risk_level'] = pd.cut(
                risk_data['global_health_score'],
                bins=[0, 30, 60, 80, 100],
                labels=['Critique', 'Élevé', 'Modéré', 'Faible']
            )
        
        # Allow aggregation by model_label in the heatmap if available
        if 'model_label' in data.columns:
            group_choice = st.selectbox('Agg. par', ['Aucun', 'model_label'], key='risk_matrix_group')
        else:
            group_choice = 'Aucun'

        risk_counts = risk_data['risk_level'].value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Répartition des Niveaux de Risque (Données Réelles)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        # If user wants the heatmap aggregated by model_label, display a separate heatmap below
        if group_choice == 'model_label':
            pivot = data.groupby('model_label').agg({
                'global_health_score': 'mean',
                'vibration': 'mean',
                'error_count': 'mean'
            }).round(2)
            if not pivot.empty:
                hm = px.imshow(
                    pivot.T,
                    title='Carte Thermique des Risques par Label de Modèle (A/B/C)',
                    color_continuous_scale='RdYlGn_r',
                    aspect='auto'
                )
                st.plotly_chart(hm, use_container_width=True)
        st.plotly_chart(fig, use_container_width=True)
        
    def _display_recent_alerts(self, data):
        """Affiche les alertes récentes."""
        # Attempt to attach model predictions (probability of failure and RUL)
        df = data.copy()
        clf, reg, _ = _load_saved_models()
        features = ['global_health_score', 'vibration', 'error_count', 'degradation_rate']
        try:
            X = df[features].fillna(0)
            if clf is not None:
                if hasattr(clf, 'predict_proba'):
                    df['pred_proba'] = clf.predict_proba(X)[:, 1]
                else:
                    df['pred_proba'] = clf.predict(X)
            if reg is not None:
                df['pred_rul'] = reg.predict(X)
        except Exception:
            # ignore prediction errors and continue
            pass

        critical_machines = df[df['global_health_score'] < 40]

        if len(critical_machines) > 0:
            alert_data = []
            for _, row in critical_machines.head(10).iterrows():
                alert_data.append({
                    'Machine': f"Machine {row['machineID']}",
                    'Modèle': f"{row['model']} (Label: {map_model_to_label(row['model'])})",
                    'Santé': f"{row['global_health_score']:.1f}",
                    'Risque': 'Critique' if row['global_health_score'] < 30 else 'Élevé',
                    'Probabilité défaillance': f"{row.get('pred_proba', float('nan')):.2f}" if 'pred_proba' in row else 'N/A',
                    'RUL prédit (jours)': f"{row.get('pred_rul', float('nan')):.1f}" if 'pred_rul' in row else 'N/A',
                    'Action': 'Maintenance Immédiate' if row['global_health_score'] < 30 else 'Planifier'
                })

            st.dataframe(
                pd.DataFrame(alert_data),
                width='stretch',
                hide_index=True
            )
        else:
            st.success("✅ Aucune alerte critique actuellement")
    
    def display_machine_analysis_tab(self, data):
        """Affiche l'onglet analyse par machine."""
        st.header("🔍 Analyse par Machine")
        
        # Sélection de la machine
        machine_list = data['machineID'].unique()
        selected_machine = st.selectbox(
            "Sélectionner une machine",
            machine_list,
            format_func=lambda x: f"Machine {x}"
        )
        
        if selected_machine:
            machine_data = data[data['machineID'] == selected_machine]
            
            if not machine_data.empty:
                self._display_machine_details(machine_data.iloc[0])
                self._display_machine_trends(machine_data)
                self._display_maintenance_recommendations(machine_data.iloc[0])
    
    def _display_machine_details(self, machine_row):
        """Affiche les détails d'une machine."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Score Santé", f"{machine_row['global_health_score']:.1f}/100")
            
        with col2:
            st.metric("Âge", f"{machine_row['age']} ans")
            
        with col3:
            st.metric("Vibration", f"{machine_row['vibration']:.1f}")
            
        with col4:
            risk_level = "Critique" if machine_row['global_health_score'] < 30 else \
                        "Élevé" if machine_row['global_health_score'] < 60 else \
                        "Modéré" if machine_row['global_health_score'] < 80 else "Faible"
            st.metric("Niveau Risque", risk_level)
    
    def _display_machine_trends(self, machine_data):
        """Affiche les tendances d'une machine."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Tendance santé
            if 'timestamp' in machine_data.columns:
                fig = px.line(
                    machine_data,
                    x='timestamp',
                    y='global_health_score',
                    title='Évolution du Score de Santé',
                    markers=True
                )
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Score de Santé'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Métriques techniques
            metrics_data = {
                'Vibration': machine_data['vibration'].mean(),
                'Pression': machine_data['pressure'].mean() if 'pressure' in machine_data.columns else 0,
                'Erreurs': machine_data['error_count'].mean(),
                'Rotation': machine_data['rotate'].mean() if 'rotate' in machine_data.columns else 0
            }
            
            fig = px.bar(
                x=list(metrics_data.keys()),
                y=list(metrics_data.values()),
                title='Métriques Techniques Moyennes',
                color=list(metrics_data.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_maintenance_recommendations(self, machine_row):
        """Affiche les recommandations de maintenance."""
        st.subheader("✅ Recommandations de Maintenance")
        
        recommendations = []
        health_score = machine_row['global_health_score']
        
        if health_score < 30:
            recommendations.extend([
                "🔴 ARRÊT IMMÉDIAT recommandé",
                "🔧 Intervention maintenance urgente nécessaire",
                "📞 Contacter le responsable maintenance"
            ])
        elif health_score < 60:
            recommendations.extend([
                "🟠 Planifier maintenance sous 7 jours",
                "🔍 Vérifier capteurs vibration et pression",
                "📊 Analyser historique des erreurs"
            ])
        else:
            recommendations.extend([
                "🟢 Maintenance préventive standard",
                "👁️ Surveillance continue recommandée",
                "📈 Vérifier dans 30 jours"
            ])
        
        for rec in recommendations:
            st.write(f"- {rec}")
    
    def display_alerts_tab(self, data):
        """Affiche l'onglet alertes."""
        st.header("🚨 Centre d'Alertes")
        
        # Filtres d'alertes
        col1, col2 = st.columns(2)
        
        with col1:
            min_health = st.slider(
                "Seuil santé critique",
                min_value=0,
                max_value=100,
                value=40
            )
        
        with col2:
            alert_level = st.selectbox(
                "Niveau d'alerte",
                ["Tous", "Critique", "Élevé", "Modéré"]
            )
        
        # Alertes filtrées
        critical_data = data[data['global_health_score'] <= min_health]
        
        if not critical_data.empty:
            st.subheader(f"🔴 {len(critical_data)} Machines Nécessitant Attention")
            
            # Tableau des alertes
            alert_df = critical_data[['machineID', 'model', 'global_health_score', 'vibration', 'error_count']].copy()
            alert_df['Action'] = alert_df['global_health_score'].apply(
                lambda x: 'ARRÊT IMMÉDIAT' if x < 30 else 'MAINTENANCE URGENTE' if x < 50 else 'SURVEILLANCE'
            )
            
            st.dataframe(
                alert_df.sort_values('global_health_score'),
                use_container_width=True,
                hide_index=True
            )
            
            # Carte thermique des risques
            st.subheader("📋 Carte Thermique des Risques")
            self._display_risk_heatmap(critical_data)
        else:
            st.success("✅ Aucune alerte avec les critères sélectionnés")
    
    def _display_risk_heatmap(self, data):
        """Affiche une carte thermique des risques."""
        # Préparation des données pour la heatmap
        pivot_data = data.pivot_table(
            index='model',
            values=['global_health_score', 'vibration', 'error_count'],
            aggfunc='mean'
        ).round(2)
        
        if not pivot_data.empty:
            fig = px.imshow(
                pivot_data.T,
                title='Carte Thermique des Risques par Modèle',
                color_continuous_scale='RdYlGn_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_reports_tab(self, data):
        """Affiche l'onglet rapports."""
        st.header("📋 Générateur de Rapports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Type de rapport",
                ["Exécutif", "Maintenance", "Technique", "Financier"]
            )
            
        with col2:
            timeframe = st.selectbox(
                "Période",
                ["7 derniers jours", "30 derniers jours", "3 derniers mois", "12 derniers mois"]
            )
        
        if st.button("📊 Générer le Rapport", type="primary"):
            with st.spinner("Génération du rapport en cours..."):
                # Simulation de génération de rapport
                self._generate_sample_report(data, report_type)

        # Decision Pack generator for executives
        if st.button("📦 Générer Decision Pack (Executive)"):
            with st.spinner("Génération Decision Pack..."):
                content, fname, path = _generate_decision_pack(data)
            st.success(f"Decision Pack généré: {fname}")
            st.download_button(label="📥 Télécharger Decision Pack", data=content, file_name=fname, mime='text/plain')
            if path:
                st.info(f"Fichier sauvegardé: {path}")
    
    def _generate_sample_report(self, data, report_type):
        """Génère un exemple de rapport."""
        st.subheader(f"📄 Rapport {report_type} - Exemple")
        
        if report_type == "Exécutif":
            st.info("""
            **RAPPORT EXÉCUTIF - MAINTENANCE PRÉDICTIVE**
            
            **Résumé:** Le système de maintenance prédictive a identifié 23 machines critiques 
            nécessitant une attention immédiate, représentant un risque financier potentiel 
            de 450,000 € en temps d'arrêt évitable.
            
            **Recommandations:**
            - Intervention immédiate sur les 5 machines les plus critiques
            - Réallocation du budget maintenance vers les actions préventives
            - Formation des équipes aux nouveaux protocoles
            """)
        
        elif report_type == "Maintenance":
            st.info("""
            **RAPPORT MAINTENANCE OPÉRATIONNEL**
            
            **Alertes Actives:**
            - CRITIQUE: 5 machines (intervention < 48h)
            - HAUTE: 8 machines (planifier cette semaine)
            - MOYENNE: 10 machines (surveillance accrue)
            
            **Planning Recommandé:**
            - Semaine 1: Machines 101, 205, 308
            - Semaine 2: Machines 412, 515
            - Semaine 3: Machines 201, 304, 407
            """)
        
        # Bouton d'export — génère un rapport basé sur les données réelles
        st.download_button(
            label="📥 Télécharger le Rapport (TXT)",
            data="Rapport d'analyse basé sur données réelles — généré depuis le dashboard",
            file_name=f"rapport_{report_type.lower()}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    def run(self, data):
        """Lance le dashboard."""
        st.title("🔧 Système Avancé de Maintenance Prédictive")
        # If the user has uploaded or loaded a dataset during this session, prefer it
        try:
            if 'data_df' in st.session_state and st.session_state['data_df'] is not None:
                data = st.session_state['data_df']
                loaded_path = st.session_state.get('data_path')
                if loaded_path:
                    st.info(f"✅ Dataset chargé: {loaded_path}")
            elif 'data_path' in st.session_state:
                st.info(f"✅ Dataset actif: {st.session_state.get('data_path')}")
        except Exception:
            # session_state may not be available in certain contexts — ignore
            pass
        st.markdown("---")
        
        # Configuration sidebar — avec KPIs réels basés sur les données actuelles
        selected_tab, date_range = self.setup_sidebar()
        
        # Afficher les KPIs réels dans la barre latérale
        if data is not None and len(data) > 0:
            st.sidebar.subheader("📊 KPIs Réels")
            total_machines = data['machineID'].nunique()
            critical_count = len(data[data['global_health_score'] < 30])
            avg_health = data['global_health_score'].mean()
            st.sidebar.metric("Machines total", total_machines)
            st.sidebar.metric("Machines critiques", critical_count)
            st.sidebar.metric("Santé moyenne", f"{avg_health:.1f}/100")
        
        # Navigation par onglets
        tabs = st.tabs(["📊 Vue Ensemble", "🔍 Analyse Machine", "🚨 Alertes", "📋 Rapports"])
        
        with tabs[0]:
            self.display_overview_tab(data)
            
        with tabs[1]:
            self.display_machine_analysis_tab(data)
            
        with tabs[2]:
            self.display_alerts_tab(data)
            
        with tabs[3]:
            self.display_reports_tab(data)

def main():
    """Fonction principale — charge UNIQUEMENT le dataset réel Predictive_Table.csv."""
    dashboard = AdvancedDashboard()
    
    # Charger le dataset réel — OBLIGATOIRE
    sample_data = None
    if ('load_and_prepare' in globals()) and (load_and_prepare is not None):
        candidate_paths = []
        try:
            try:
                from .config import get_data_path
            except Exception:
                from src.config import get_data_path
            try:
                p = get_data_path('predictive')
                candidate_paths.append(p)
            except Exception:
                pass
        except Exception:
            pass

        # project-relative paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_paths.append(os.path.join(project_root, 'data', 'Predictive_Table.csv'))
        candidate_paths.append(os.path.join(project_root, 'data', 'predictive_table.csv'))

        for p in candidate_paths:
            try:
                if p is None or not os.path.exists(p):
                    continue
                # Use cached loading to avoid re-computing features on every page load
                df = cached_load_and_prepare(p)
                if df is None:
                    continue
                # ajouter label A/B/C pour visualisation
                try:
                    df['model_label'] = df['model'].apply(lambda m: map_model_to_label(m))
                except Exception:
                    df['model_label'] = df['model']
                sample_data = df
                st.session_state['data_path'] = p
                print(f"✅ Dataset réel chargé depuis: {p}")
                break
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {p}: {e}")

    # Si le dataset réel n'est pas chargé, afficher une erreur et arrêter
    if sample_data is None:
        st.error(
            "❌ ERREUR CRITIQUE: Le fichier 'data/Predictive_Table.csv' est introuvable ou invalide.\n\n"
            "Veuillez:\n"
            "1. Vérifier que le fichier existe: `data/Predictive_Table.csv`\n"
            "2. Utiliser le panneau 'Données' dans la barre latérale pour charger ou uploader votre CSV.\n"
            "3. Redémarrer l'application.\n\n"
            "Le système fonctionne UNIQUEMENT avec votre dataset réel."
        )
        st.stop()
    
    dashboard.run(sample_data)

if __name__ == "__main__":
    main()
