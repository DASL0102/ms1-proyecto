import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =======================================================
# Parámetros del dataset sintético reducido
# =======================================================
days = 1  # cantidad de días
patients_per_day = 20  # solo 6 pacientes por día
start_date = datetime(2025, 8, 1)

categories = ['Verde', 'Amarilla', 'Roja']
mean_triage = {'Verde': 5, 'Amarilla': 8, 'Roja': 3}
mean_service = {'Verde': 15, 'Amarilla': 20, 'Roja': 45}
p_derived = {'Verde': 0.01, 'Amarilla': 0.10, 'Roja': 0.30}
p_balk = {'Verde': 0.05, 'Amarilla': 0.02, 'Roja': 0.00}
mean_transfer = 30

# Horario de médicos por hora (coherente)
doctors_by_hour = {}
for h in range(24):
    if 0 <= h < 6:
        doctors_by_hour[h] = 1
    elif 6 <= h < 14:
        doctors_by_hour[h] = 2
    elif 14 <= h < 22:
        doctors_by_hour[h] = 3
    else:
        doctors_by_hour[h] = 1

# =======================================================
# Generación de registros
# =======================================================
rows = []
patient_id = 1
np.random.seed(42)

for day in range(days):
    current_date = start_date + timedelta(days=day)
    
    for _ in range(patients_per_day):
        hour = np.random.randint(0, 24)
        category = np.random.choice(categories)
        doctors = doctors_by_hour[hour]
        
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        arrival_dt = datetime(current_date.year, current_date.month, current_date.day,
                              hour, minute, second)
        
        # Triagem tiempo
        triage_time = np.random.exponential(mean_triage[category])
        
        # Decidir si paciente acepta esperar (balked)
        balked = int(np.random.rand() < p_balk[category])
        
        # Tiempo de servicio y derivación coherentes
        if balked or doctors == 0:
            service_time = ''
            derived = 0
            transfer_time = ''
        else:
            service_time = np.random.exponential(mean_service[category])
            derived = int(np.random.rand() < p_derived[category])
            transfer_time = np.random.exponential(mean_transfer) if derived else ''
        
        rows.append({
            'patient_id': patient_id,
            'arrival_datetime': arrival_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'category': category,
            'triage_time_min': round(float(triage_time), 12),
            'service_time_min': round(float(service_time), 12) if service_time != '' else '',
            'derived_intrinsic': derived,
            'transfer_time_min': round(float(transfer_time), 12) if transfer_time != '' else '',
            'balked': balked,
            'doctors_on_duty': doctors,
            'arrival_date': arrival_dt.strftime('%Y-%m-%d'),
            'arrival_time': arrival_dt.strftime('%H:%M:%S')
        })
        patient_id += 1

# Convertir a DataFrame y guardar CSV
df = pd.DataFrame(rows)
out_path = "camiula_sintetico_10pacientes.csv"
df.to_csv(out_path, index=False)
out_path
