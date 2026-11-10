"""Cuenta, solo con EXPLAIN (sin ejecutar), cuántos planes cambian respecto de PostgreSQL para cada archivo de predicciones y semilla."""
import csv, json, os, sys
import psycopg2
sys.path.insert(0, "/root/exp")
from importlib import import_module
firma = import_module("03_ejecutar").firma

D = "/root/exp/datos"
meta = list(csv.DictReader(open(f"{D}/subplanes_meta.csv")))
completas = {m["consulta"]: m for m in meta if m["es_completa"] == "1"}
reales = {str(i): int(l.split(",")[-1]) for i, l in enumerate(open(f"{D}/subplanes.csv"))}
hints = {"pg": {}, "reales": {}}
for m in meta:
    hints["reales"].setdefault(m["consulta"], []).append((m["alias"], reales[m["fila"]]))
archivos = {"cuantico": f"{D}/predicciones_cuantico.csv", "clasico": f"{D}/predicciones_clasico.csv",
            "cuantico_sinsep": f"{D}/predicciones_cuantico_sin_separacion.csv",
            "cuantico_sinsep_2000": "/root/sens/predicciones_cuantico_sin_separacion.csv"}
for nombre, ruta in archivos.items():
    if not os.path.exists(ruta):
        continue
    for r in csv.DictReader(open(ruta)):
        hints.setdefault(f"{nombre}_s{r['semilla']}", {}).setdefault(r["consulta"], []).append((r["alias"], int(r["filas_corregidas"])))
con = psycopg2.connect(dbname="imdb"); con.autocommit = True; cur = con.cursor()
cur.execute("LOAD 'pg_hint_plan'"); cur.execute("SET max_parallel_workers_per_gather = 0")
firmas = {}
for cfg, h in hints.items():
    for q, m in completas.items():
        lista = h.get(q, [])
        bloque = "/*+ " + " ".join(f"Rows({a} #{n})" for a, n in lista) + " */ " if lista else ""
        cur.execute(bloque + "EXPLAIN (FORMAT JSON) " + m["sql"])
        firmas[(cfg, q)] = firma(cur.fetchone()[0][0]["Plan"])
res = {cfg: sorted((q for q in completas if firmas[(cfg, q)] != firmas[("pg", q)]), key=int) for cfg in hints}
json.dump(res, open("/root/sens/planes_explain.json", "w"), indent=1)
for cfg, qs in sorted(res.items()):
    print(f"{cfg:24s} {len(qs):3d} planes distintos: {' '.join(qs)}")
