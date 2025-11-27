"""
proyecto2_equipo_corregido.py
Modelo de simulación del proceso de urgencias CAMIULA usando SimPy.
"""

import simpy
import pandas as pd
import numpy as np

# =======================================================
# 1. CONFIGURACIÓN FLEXIBLE
# =======================================================

CONFIG = {
    "use_csv": True,                     # usar datos reales
    "csv_path": "camiula_sintetico_10pacientes.csv",      # ruta del CSV
    "override_doctors": None,            # ej: 3 para forzar 3 médicos
    "override_triage_time": None,        # ej: lambda: np.random.exponential(2)
    "override_service_time": None,       # ej: lambda cat: np.random.exponential(5)
    "override_transfer_prob": 0.50,      # ej: 0.15
    "verbose": True
}
# CONFIG = {
#     "use_csv": True,                     # usar datos reales
#     #"csv_path": "camiula_sintetico_10pacientes.csv",      # ruta del CSV
#     "override_doctors": 10,            # ej: 3 para forzar 3 médicos
#     "override_triage_time": None,        # ej: lambda: np.random.exponential(2)
#     "override_service_time": None,       # ej: lambda cat: np.random.exponential(5)
#     "override_transfer_prob": None,      # ej: 0.15
#     "verbose": True
# }

#RUN_TIME = 3*24*60
RUN_TIME = 24*60


# =======================================================
# 2. CARGA DE DATOS
# =======================================================

def load_data(config):
    if not config["use_csv"]:
        print("[INFO] No se usará CSV. Se usarán parámetros fijos.")
        return None

    try:
        df = pd.read_csv(config["csv_path"])
        print(f"[INFO] CSV cargado con {len(df)} registros.")

        # Limpiar y completar valores faltantes sin warnings
        df["triage_time_min"] = df["triage_time_min"].fillna(
            df["triage_time_min"].mean() if df["triage_time_min"].notnull().any() else 2
        )
        df["service_time_min"] = df["service_time_min"].fillna(
            df["service_time_min"].mean() if df["service_time_min"].notnull().any() else 5
        )
        df["derived_intrinsic"] = df["derived_intrinsic"].fillna(0.1)
        df["doctors_on_duty"] = df["doctors_on_duty"].fillna(2)

        # Convertir tiempo de llegada a minutos desde el inicio del día
        # Convertir tiempo de llegada a minutos desde el primer paciente
        df["arrival_min"] = (
            pd.to_datetime(df["arrival_datetime"], errors='coerce') - 
            pd.to_datetime(df["arrival_datetime"], errors='coerce').min()
        ).dt.total_seconds() / 60
        df["arrival_min"].fillna(0, inplace=True)


        return df

    except Exception as e:
        print("[ERROR] No se pudo leer el CSV:", e)
        return None

# =======================================================
# 3. PROCESOS DE SIMULACIÓN
# =======================================================

class CamiulaSystem:

    def __init__(self, env, num_doctors=2):
        self.env = env
        self.doctors = simpy.Resource(env, capacity=num_doctors)

        # Métricas
        self.wait_times = []
        self.service_times = []
        self.derived_count = 0
        self.attended_count = 0

    # -------------------------
    # Proceso principal paciente
    # -------------------------
    def patient_process(self, patient, triage_time, service_time, derive_prob):

        arrival = self.env.now

        # TRIAGE
        yield self.env.timeout(triage_time)

        # COLA PARA MÉDICO
        with self.doctors.request() as req:
            yield req
            wait = self.env.now - arrival
            self.wait_times.append(wait)

            # ATENCIÓN
            yield self.env.timeout(service_time)

            # derivación
            if np.random.rand() < derive_prob:
                self.derived_count += 1
            else:
                self.attended_count += 1

# =======================================================
# 4. GENERADOR DE PACIENTES
# =======================================================

def generate_from_csv(env, system, df, config):
    for _, row in df.iterrows():

        # esperar hasta llegada
        yield env.timeout(row["arrival_min"] - env.now if row["arrival_min"] > env.now else 0)

        triage_time = row["triage_time_min"] if not config["override_triage_time"] else config["override_triage_time"]()
        service_time = row["service_time_min"] if not config["override_service_time"] else config["override_service_time"](row.get("category", None))
        derive_prob = row["derived_intrinsic"] if not config["override_transfer_prob"] else config["override_transfer_prob"]

        env.process(system.patient_process(
            patient=row.get("patient_id", "unknown"),
            triage_time=triage_time,
            service_time=service_time,
            derive_prob=derive_prob
        ))

# def generate_synthetic(env, system, arrival_rate=10):
#     i = 0
#     while True:
#         yield env.timeout(np.random.exponential(60 / arrival_rate))
#         i += 1
#         triage_time = np.random.exponential(2)
#         service_time = np.random.exponential(6)
#         derive_prob = 0.10

#         env.process(system.patient_process(
#             patient=i,
#             triage_time=triage_time,
#             service_time=service_time,
#             derive_prob=derive_prob
#         ))

def generate_synthetic(env, system, arrival_rate_by_hour=None):
    """
    Genera pacientes sintéticos, pudiendo definir tasas de llegada por hora.
    arrival_rate_by_hour: lista de 24 valores (pacientes por hora)
    """
    i = 0
    while True:
        current_hour = int(env.now // 60) % 24  # hora simulada
        rate = arrival_rate_by_hour[current_hour] if arrival_rate_by_hour else 10
        yield env.timeout(np.random.exponential(60 / rate))
        i += 1
        # tiempos más realistas
        categories = ["Verde","Amarilla","Roja"]
        prob_derive = {"Verde":0.05,"Amarilla":0.1,"Roja":0.2}
        cat = np.random.choice(categories, p=[0.5,0.3,0.2])
        derive_prob = prob_derive[cat]
        triage_time = np.random.exponential({"Verde":3,"Amarilla":5,"Roja":8}[cat])
        service_time = np.random.exponential({"Verde":10,"Amarilla":20,"Roja":40}[cat])
        triage_time = max(1, triage_time)  # no negativo
        service_time = max(1, service_time)
        derive_prob = 0.1  # puedes hacer que dependa de categoría

        env.process(system.patient_process(
            patient=i,
            triage_time=triage_time,
            service_time=service_time,
            derive_prob=derive_prob
        ))


# =======================================================
# 5. FUNCIÓN PRINCIPAL DE SIMULACIÓN
# =======================================================

def run_simulation(config=CONFIG, sim_time=RUN_TIME):
    df = load_data(config)

    if config["override_doctors"]:
        num_doctors = config["override_doctors"]
    elif df is not None:
        num_doctors = int(df["doctors_on_duty"].mode()[0])
    else:
        num_doctors = 2

    env = simpy.Environment()
    system = CamiulaSystem(env, num_doctors=num_doctors)

    if df is not None:
        env.process(generate_from_csv(env, system, df, config))
    else:
        env.process(generate_synthetic(env, system, arrival_rate_by_hour = [5]*6 + [15]*6 + [10]*6 + [20]*6))

    env.run(until=sim_time)

    return {
        "wait_avg": float(np.mean(system.wait_times)) if system.wait_times else 0,
        "wait_max": float(np.max(system.wait_times)) if system.wait_times else 0,
        "attended": system.attended_count,
        "derived": system.derived_count,
        "total": system.attended_count + system.derived_count
    }

# =======================================================
# 6. EJEMPLO DE USO
# =======================================================

if __name__ == "__main__":
    results = run_simulation()
    print("===== RESULTADOS =====")
    print(results)
