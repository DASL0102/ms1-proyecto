"""
sim_camiula_full.py

SimPy + Lectura CSV + Métricas reales + Resultados teóricos M/M/c/K y M/M/c
"""

import simpy
import pandas as pd
import numpy as np
import math
from typing import Dict, Any
import datetime

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
# CONFIG = {
#     "use_csv": True,
#     "csv_path": "camiula_sintetico_10pacientes.csv",
#     "override_doctors": None,
#     "override_triage_time": None,
#     "override_service_time": None,
#     "override_transfer_prob": None,
#     "verbose": True
# }

CONFIG = {
    "use_csv": True,                     # usar datos reales
    "csv_path": "camiula_sintetico_generado.csv",      # ruta del CSV
    "override_doctors": None,            # ej: 3 para forzar 3 médicos
    "override_triage_time": None,        # ej: lambda: np.random.exponential(2)
    "override_service_time": None,       # ej: lambda cat: np.random.exponential(5)
    "override_transfer_prob": None,      # ej: 0.15
    "verbose": True
}


RUN_TIME = 24 * 60  # 24h en minutos


# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    if not config["use_csv"]:
        return None

    try:
        df = pd.read_csv(config["csv_path"])
        if config["verbose"]:
            print(f"[INFO] CSV cargado ({len(df)} filas): {config['csv_path']}")

        df["triage_time_min"] = df.get("triage_time_min", pd.Series()).fillna(2)
        df["service_time_min"] = df.get("service_time_min", pd.Series()).fillna(5)
        df["derived_intrinsic"] = df.get("derived_intrinsic", pd.Series()).fillna(0.1)
        df["doctors_on_duty"] = df.get("doctors_on_duty", pd.Series()).fillna(2)

        if "arrival_datetime" in df:
            times = pd.to_datetime(df["arrival_datetime"], errors="coerce")
            min_t = times.min()
            df["arrival_min"] = (times - min_t).dt.total_seconds() / 60.0
            df["arrival_min"].fillna(0, inplace=True)
        elif "arrival_min" not in df:
            df["arrival_min"] = np.arange(len(df)).astype(float)

        return df

    except Exception as e:
        print("[ERROR] No se pudo cargar CSV:", e)
        return None


# ---------------------------------------------------------
# SISTEMA SIMPY
# ---------------------------------------------------------
class CamiulaSystem:
    def __init__(self, env: simpy.Environment, num_doctors: int):
        self.env = env
        self.doctors = simpy.Resource(env, capacity=num_doctors)

        self.arrival_times = []
        self.start_service_times = []
        self.end_service_times = []
        self.service_times = []
        self.wait_times = []

        self.attended_count = 0
        self.derived_count = 0

    def patient_process(self, patient_id, triage_time, service_time, derive_prob):
        arrival = self.env.now
        self.arrival_times.append(arrival)

        yield self.env.timeout(triage_time)

        with self.doctors.request() as req:
            yield req

            start = self.env.now
            self.start_service_times.append(start)

            wait = start - arrival
            self.wait_times.append(wait)

            yield self.env.timeout(service_time)

            end = self.env.now
            self.end_service_times.append(end)
            self.service_times.append(service_time)

            if np.random.rand() < derive_prob:
                self.derived_count += 1
            else:
                self.attended_count += 1


# ---------------------------------------------------------
# GENERADORES
# ---------------------------------------------------------
def generate_from_csv(env: simpy.Environment, system: CamiulaSystem, df: pd.DataFrame, config):
    for _, row in df.iterrows():
        arrival_min = float(row["arrival_min"])
        yield env.timeout(max(arrival_min - env.now, 0.0))

        triage_time = row["triage_time_min"] if config["override_triage_time"] is None else config["override_triage_time"]()
        service_time = row["service_time_min"] if config["override_service_time"] is None else config["override_service_time"](row.get("category", None))
        derive_prob = row["derived_intrinsic"] if config["override_transfer_prob"] is None else config["override_transfer_prob"]

        env.process(system.patient_process(
            patient_id=row.get("patient_id", None),
            triage_time=float(triage_time),
            service_time=float(service_time),
            derive_prob=float(derive_prob)
        ))




def generate_synthetic(env, system, arrival_rate_by_hour=None, csv_path="camiula_sintetico_generado.csv"):
    i = 0
    synthetic_store = []

    # hora inicial simulada como "hoy" desde medianoche
    start_datetime = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    max_minutes = 24 * 60  # un día en minutos

    while env.now < max_minutes:
        current_hour = int(env.now // 60) % 24
        rate = arrival_rate_by_hour[current_hour] if arrival_rate_by_hour else 10
        yield env.timeout(np.random.exponential(60 / rate))

        i += 1
        categories = ["Verde", "Amarilla", "Roja"]
        prob_derive = {"Verde": 0.05, "Amarilla": 0.1, "Roja": 0.2}
        cat = np.random.choice(categories, p=[0.5, 0.3, 0.2])
        derive_prob = prob_derive[cat]

        triage_time = max(1, np.random.exponential({"Verde":3,"Amarilla":5,"Roja":8}[cat]))
        service_time = max(1, np.random.exponential({"Verde":10,"Amarilla":20,"Roja":40}[cat]))

        arrival_datetime = start_datetime + datetime.timedelta(minutes=env.now)
        arrival_date = arrival_datetime.date().isoformat()
        arrival_time = arrival_datetime.time().strftime("%H:%M:%S")

        row = {
            "patient_id": i,
            "arrival_datetime": arrival_datetime.isoformat(),
            "hour": current_hour,
            "category": cat,
            "triage_time_min": triage_time,
            "service_time_min": service_time,
            "derived_intrinsic": derive_prob,
            "transfer_time_min": np.nan,
            "balked": 0,
            "doctors_on_duty": system.doctors.capacity,
            "arrival_date": arrival_date,
            "arrival_time": arrival_time
        }

        synthetic_store.append(row)

        env.process(system.patient_process(
            patient_id=i,
            triage_time=triage_time,
            service_time=service_time,
            derive_prob=derive_prob
        ))

        pd.DataFrame(synthetic_store).to_csv(csv_path, index=False)


# ---------------------------------------------------------
# CORRER SIMULACIÓN
# ---------------------------------------------------------
def run_simulation(config=CONFIG, sim_time=RUN_TIME):

    df = load_data(config)

    if config["override_doctors"] is not None:
        num_doctors = int(config["override_doctors"])
    else:
        num_doctors = int(df["doctors_on_duty"].mode()[0]) if df is not None else 2

    env = simpy.Environment()
    system = CamiulaSystem(env, num_doctors=num_doctors)

    if df is not None:
        env.process(generate_from_csv(env, system, df, config))
    else:
        env.process(generate_synthetic(env, system, arrival_rate_by_hour = [5]*6 + [15]*6 + [10]*6 + [20]*6))

    env.run(until=sim_time)

    wait_avg = float(np.mean(system.wait_times)) if system.wait_times else 0.0
    wait_max = float(np.max(system.wait_times)) if system.wait_times else 0.0
    attended = system.attended_count
    derived = system.derived_count
    total = attended + derived

    legacy = {
        "wait_avg": wait_avg,
        "wait_max": wait_max,
        "attended": attended,
        "derived": derived,
        "total": total
    }

    detailed = {
        "arrival_times": system.arrival_times,
        "start_service_times": system.start_service_times,
        "end_service_times": system.end_service_times,
        "service_times": system.service_times,
        "wait_times": system.wait_times,
        "num_doctors": num_doctors
    }

    return {"legacy": legacy, "detailed": detailed}


# ---------------------------------------------------------
# MÉTRICAS λ y μ
# ---------------------------------------------------------
def calcular_tasas(detailed, horas):
    arr = np.array(detailed["arrival_times"])
    serv = np.array(detailed["service_times"])

    lmbda = len(arr) / horas
    mean_service_min = np.mean(serv) if len(serv) else 1
    mu = 1 / (mean_service_min / 60.0)

    # 👇 ESTO ES EXACTAMENTE LO QUE ME PEDISTE
    arrival_diffs = np.diff(arr) if len(arr) > 1 else [0]
    prom_llegadas = float(np.mean(arrival_diffs))
    prom_servicio = float(mean_service_min)

    extras = {
        "promedio_tiempos_entre_llegadas_min": prom_llegadas,
        "promedio_servicio_min": prom_servicio,
        "tasa_llegada_por_hora": lmbda,
        "tasa_servicio_por_hora": mu,
        "clientes": len(arr)
    }

    print("\n--- MÉTRICAS ADICIONALES (pedidas) ---")
    print(extras)

    return lmbda, mu


# ---------------------------------------------------------
# MODELOS TEÓRICOS
# ---------------------------------------------------------
def mmck_teorico(lmbda, mu, c, K):
    rho = lmbda / (c * mu)

    sum1 = sum((lmbda/mu)**n / math.factorial(n) for n in range(c))
    sum2 = ((lmbda/mu)**c / math.factorial(c)) * ((1 - rho**(K - c + 1)) / (1 - rho))
    P0 = 1.0 / (sum1 + sum2)

    PK = ((lmbda/mu)**K / (math.factorial(c) * c**(K - c))) * P0

    if rho != 1:
        Lq = (
            P0 * ((lmbda/mu)**c * rho / (math.factorial(c) * (1 - rho)**2))
            * (1 - rho**(K - c + 1) - (K - c + 1)*(1 - rho)*rho**(K - c))
        )
    else:
        Lq = 0

    L = Lq + (lmbda * (1 - PK) / mu)

    Wq = Lq / (lmbda * (1 - PK)) if lmbda*(1-PK) > 0 else 0
    W = Wq + 1/mu

    return {
        "Wq_h": Wq/60,
        "W_h": W/60,
        "Lq": Lq,
        "L": L,
        "rho_percent": rho*100,
        "idle_percent": (1-rho)*100,
        "lost_capacity_total": PK * lmbda,
        "lost_capacity_percent": PK*100
    }


def mmc_teorico(lmbda, mu, c):
    rho = lmbda / (c * mu)

    sum1 = sum((lmbda/mu)**n / math.factorial(n) for n in range(c))
    sum2 = ((lmbda/mu)**c / math.factorial(c)) * (1 / (1 - rho))
    P0 = 1.0 / (sum1 + sum2)

    Lq = (P0 * (lmbda/mu)**c * rho) / (math.factorial(c) * (1 - rho)**2)
    L = Lq + lmbda/mu

    Wq = Lq / lmbda
    W = Wq + 1/mu

    return {
        "Wq_h": Wq/60,
        "W_h": W/60,
        "Lq": Lq,
        "L": L,
        "rho_percent": rho*100,
        "idle_percent": (1-rho)*100,
        "lost_capacity_total": 0,
        "lost_capacity_percent": 0
    }


# ---------------------------------------------------------
# REPORTES TEÓRICOS
# ---------------------------------------------------------
def generar_reportes_teoricos(detailed, horas, K=200):
    lmbda, mu = calcular_tasas(detailed, horas)
    c = detailed["num_doctors"]

    return (
        mmck_teorico(lmbda, mu, c, K),
        mmc_teorico(lmbda, mu, c)
    )



def imprimir_resultados_descriptivos(titulo, datos):
    print("\n==================================================")
    print(f"--- {titulo} ---")
    print("==================================================")

    nombres = {
        "Wq_h": "Tiempo Promedio de Espera en Cola (horas)",
        "W_h": "Tiempo Promedio Total en el Sistema (horas)",
        "Lq": "Clientes Promedio en Cola",
        "L": "Clientes Promedio en el Sistema",
        "rho_percent": "Porcentaje de Ocupación de los Médicos (%)",
        "idle_percent": "Porcentaje de Inactividad de los Médicos (%)",
        "lost_capacity_total": "Clientes Perdidos por Capacidad (total)",
        "lost_capacity_percent": "Clientes Perdidos por Capacidad (%)"
    }

    for key, val in datos.items():
        nombre = nombres.get(key, key)
        print(f"{nombre:50s} {val:.4f}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== Ejecutando simulación ===")
    out = run_simulation(CONFIG, RUN_TIME)

    legacy = out["legacy"]
    detailed = out["detailed"]

    print("\n--- Resultados clásicos ---")
    print(legacy)

    teorico_mmck, teorico_mmc = generar_reportes_teoricos(detailed, horas=24)

    imprimir_resultados_descriptivos("Resultados Promediados de la Simulación (M/M/c/K)", teorico_mmck)
    imprimir_resultados_descriptivos("Resultados Teóricos (M/M/c sin capacidad K)", teorico_mmc)

