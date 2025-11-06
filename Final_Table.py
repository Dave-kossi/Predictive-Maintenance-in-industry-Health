# Conception de la table finale pour la modélisation Kaplan–Meier
import pandas as pd
import numpy as np

# === Chargement des données ===
errors = pd.read_csv('PdM_errors.csv')
failures = pd.read_csv('PdM_failures.csv')
machines = pd.read_csv('PdM_machines.csv')
maint = pd.read_csv('PdM_maint.csv')
telemetry = pd.read_csv('PdM_telemetry.csv')

# === Conversion automatique des colonnes datetime ===
for df_name, df in {'failures': failures, 'errors': errors, 'maint': maint, 'telemetry': telemetry}.items():
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        print(f"Pas de colonne 'datetime' dans {df_name}, ignorée.")

# === Détermination des bornes temporelles ===
# Début d'utilisation
start_times = telemetry.groupby('machineID')['datetime'].min().reset_index()
start_times.rename(columns={'datetime': 'start_time'}, inplace=True)

# Première panne observée (s’il y en a)
failure_time = failures.groupby('machineID')['datetime'].min().reset_index()
failure_time.rename(columns={'datetime': 'failure_time'}, inplace=True)

# Dernière mesure enregistrée
last_measure = telemetry.groupby('machineID')['datetime'].max().reset_index()
last_measure.rename(columns={'datetime': 'last_telemetry_time'}, inplace=True)

# Fusion
time_df = start_times.merge(failure_time, on='machineID', how='left')
time_df = time_df.merge(last_measure, on='machineID', how='left')

# === Création des variables time et event ===
time_df['event'] = np.where(time_df['failure_time'].notna(), 1, 0)
time_df['end_time'] = np.where(time_df['event'] == 1,
                               time_df['failure_time'],
                               time_df['last_telemetry_time'])
time_df['time'] = (time_df['end_time'] - time_df['start_time']).dt.total_seconds() / (3600*24)  # durée en jours

# === Ajout de covariables contextuelles ===
# Moyennes des mesures de télémétrie
telemetry_means = telemetry.groupby('machineID')[['volt', 'rotate', 'pressure', 'vibration']].mean().reset_index()

# Nombre d’erreurs par machine
error_count = errors.groupby('machineID').size().reset_index(name='error_count')

# Nombre de maintenances par machine
maint_count = maint.groupby('machineID').size().reset_index(name='maint_count')

# Fusion complète
final_df = time_df.merge(machines, on='machineID', how='left')
final_df = final_df.merge(telemetry_means, on='machineID', how='left')
final_df = final_df.merge(error_count, on='machineID', how='left')
final_df = final_df.merge(maint_count, on='machineID', how='left')

# Remplacement des NaN
final_df.fillna({'error_count': 0, 'maint_count': 0}, inplace=True)

# === Table finale prête pour Kaplan–Meier ===
final_df = final_df[['machineID', 'model', 'age', 'time', 'event',
                     'volt', 'rotate', 'pressure', 'vibration',
                     'error_count', 'maint_count']]

print(final_df.head())
final_df.to_csv('Predictive_Table.csv', index=False)
print("\nTable finale enregistrée sous 'Predictive_Table.csv'")
