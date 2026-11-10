import csv, json, math, sys
import numpy as np
def q(e, r): e, r = max(e, 1), max(r, 1); return max(e / r, r / e)
def resumen(ruta):
    rr = [r for r in csv.DictReader(open(ruta)) if r["semilla"] == "1"]
    cc = [r for r in rr if r["es_completa"] == "1"]
    lf = [abs(float(r["log_factor"])) for r in rr]
    qs = [q(float(r["filas_corregidas"]), float(r["filas_reales"])) for r in cc]
    qp = [q(float(r["filas_pg"]), float(r["filas_reales"])) for r in cc]
    return {"mediana": float(np.median(qs)), "p95": float(np.percentile(qs, 95)), "fmax": math.exp(max(lf)),
            "ceros": sum(x < 1e-9 for x in lf) / len(lf), "pg_mediana": float(np.median(qp))}
a = resumen("/root/exp/datos/predicciones_cuantico_sin_separacion.csv")
b = resumen("/root/sens/predicciones_cuantico_sin_separacion.csv")
px = json.load(open("/root/sens/planes_explain.json"))
p300, p2000 = len(px["cuantico_sinsep_s1"]), len(px["cuantico_sinsep_2000_s1"])
fmt = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
texto = (f"Para acotar esta amenaza se repitió el entrenamiento de QCardCorr con 2000 iteraciones, sin separar las consultas de prueba, "
         f"que es el caso más favorable para el modelo. La mediana del q-error de las consultas completas pasó de {fmt(a['mediana'])} a {fmt(b['mediana'])} "
         f"y el percentil 95 de {fmt(a['p95'],1)} a {fmt(b['p95'],1)}, el mayor factor de corrección fue de {fmt(b['fmax'])} "
         f"y el plan cambió en {p2000} consultas, frente a {p300} con 300 iteraciones.")
json.dump({"a": a, "b": b, "p300": p300, "p2000": p2000, "texto": texto}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print(json.dumps({"a": a, "b": b, "p300": p300, "p2000": p2000}, indent=1)); print(texto)
