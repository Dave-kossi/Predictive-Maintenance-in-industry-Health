# advanced_dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from advanced_predictive_model import AdvancedPredictiveMaintenance

def main():
    st.set_page_config(
        page_title="Maintenance Prédictive",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("Système de Maintenance Prédictive")
    
    # Initialisation simple
    if 'model' not in st.session_state:
        st.session_state.model = AdvancedPredictiveMaintenance()
        st.session_state.initialized = False
    
    # Chargement des données
    if not st.session_state.initialized:
        with st.spinner('Initialisation du système...'):
            if st.session_state.model.load_data():
                st.session_state.model.train_models()
                st.session_state.initialized = True
                st.success("Système prêt !")
            else:
                st.error("Erreur de chargement des données")
                return
    
    # Navigation simple
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Analyse Machine", "Courbes de Survie"])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_machine_analysis()
    
    with tab3:
        show_survival_analysis()

def show_overview():
    st.header("Vue d'ensemble du parc")
    
    model = st.session_state.model
    risk_dist = model.get_fleet_risk_distribution()
    
    # KPI simples
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Machines totales", len(model.data))
    
    with col2:
        avg_health = model.data['global_health_score'].mean()
        st.metric("Santé moyenne", f"{avg_health:.1f}/100")
    
    with col3:
        critical_count = len(risk_dist['machines_critiques'])
        st.metric("Machines critiques", critical_count)
    
    with col4:
        failure_rate = model.data['event'].mean() * 100
        st.metric("Taux de défaillance", f"{failure_rate:.1f}%")
    
    # Distribution des risques
    st.subheader("Distribution des risques")
    if risk_dist['distribution']:
        fig = px.pie(
            values=list(risk_dist['distribution'].values()),
            names=list(risk_dist['distribution'].keys()),
            title="Niveaux de risque"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Machines critiques
    if risk_dist['machines_critiques']:
        st.subheader("Machines nécessitant attention")
        critical_data = []
        for machine_id in risk_dist['machines_critiques'][:10]:  # Limiter à 10
            machine_data = model.data[model.data['machineID'] == machine_id].iloc[0]
            critical_data.append({
                'ID': machine_id,
                'Modèle': machine_data['model'],
                'Âge': machine_data['age'],
                'Santé': f"{machine_data['global_health_score']:.1f}"
            })
        
        st.dataframe(pd.DataFrame(critical_data), use_container_width=True)

def show_machine_analysis():
    st.header("Analyse par machine")
    
    model = st.session_state.model
    machine_list = model.data['machineID'].tolist()
    
    selected_machine = st.selectbox(
        "Choisir une machine",
        machine_list,
        format_func=lambda x: f"Machine {x}"
    )
    
    if st.button("Analyser"):
        with st.spinner("Analyse en cours..."):
            analysis = model.get_machine_comprehensive_analysis(selected_machine)
            
            # Affichage des résultats
            st.subheader(f"Machine {analysis['identification']['machine_id']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Score santé", f"{analysis['sante_globale']['score_global']}/100")
            
            with col2:
                st.metric("Niveau risque", analysis['prediction_risque']['risk_level'])
            
            with col3:
                st.metric("RUL médian", f"{analysis['estimation_rul']['rul_median']} jours")
            
            with col4:
                st.metric("Âge", f"{analysis['identification']['age']} ans")
            
            # Recommandations
            st.subheader("Recommandations de maintenance")
            for i, rec in enumerate(analysis['recommandations_maintenance'], 1):
                st.write(f"{i}. {rec}")
            
            # Anomalies
            anomalies = analysis['anomalies_detectees']
            if anomalies:
                st.subheader("Anomalies détectées")
                for anomaly in anomalies:
                    st.warning(f"{anomaly['type']}: {anomaly['message']}")

def show_survival_analysis():
    st.header("Analyse de survie")
    
    model = st.session_state.model
    
    # Courbe de survie
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model.kmf.timeline,
        y=model.kmf.survival_function_['KM_estimate'],
        mode='lines',
        name='Probabilité de survie'
    ))
    
    fig.update_layout(
        title='Courbe de survie Kaplan-Meier',
        xaxis_title='Temps (jours)',
        yaxis_title='Probabilité de survie'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistiques de survie")
        st.write(f"Temps médian: {model.kmf.median_survival_time_:.1f} jours")
        st.write(f"Survie à 30 jours: {model.kmf.predict(30):.2%}")
        st.write(f"Survie à 90 jours: {model.kmf.predict(90):.2%}")
        st.write(f"Survie à 1 an: {model.kmf.predict(365):.2%}")
    
    with col2:
        st.subheader("Données du parc")
        st.write(f"Machines totales: {len(model.data)}")
        st.write(f"Pannes enregistrées: {model.data['event'].sum()}")
        st.write(f"Temps moyen: {model.data['time'].mean():.1f} jours")

if __name__ == "__main__":
    main()