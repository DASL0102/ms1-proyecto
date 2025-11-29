import pandas as pd
import json

# Cargar el CSV
df = pd.read_csv("camiula_sintetico_generado.csv", sep="\t")  # Usa '\t' porque tu CSV está tabulado

# Convertir a JSON
json_data = df.to_json(orient="records", date_format="iso")

# Guardar en un archivo JSON
with open("datos.json", "w") as f:
    f.write(json_data)

# Imprimir JSON en pantalla (opcional)
print(json.dumps(json.loads(json_data), indent=4))
