import csv, sys
import psycopg2
sys.path.insert(0, "/root/exp")
from importlib import import_module
firma = import_module("03_ejecutar").firma
D = "/root/exp/datos"
meta = list(csv.DictReader(open(f"{D}/subplanes_meta.csv")))
comp = {m["consulta"]: m for m in meta if m["es_completa"] == "1"}
h = {}
for r in csv.DictReader(open("/root/sens/predicciones_cuantico_sin_separacion.csv")):
    h.setdefault(r["consulta"], []).append((r["alias"], int(r["filas_corregidas"])))
ej = {}
for r in csv.DictReader(open(f"{D}/ejecuciones.csv")):
    if r["rep"] != "0": ej.setdefault((r["consulta"], r["config"]), (r["firma_plan"], []))[1].append(float(r["ms"]))
con = psycopg2.connect(dbname="imdb"); con.autocommit = True; cur = con.cursor()
cur.execute("LOAD 'pg_hint_plan'"); cur.execute("SET max_parallel_workers_per_gather = 0")
for q in ["8", "37", "42", "47", "48", "57", "58", "59", "65"]:
    cur.execute("/*+ " + " ".join(f"Rows({a} #{n})" for a, n in h[q]) + " */ EXPLAIN (FORMAT JSON) " + comp[q]["sql"])
    f = firma(cur.fetchone()[0][0]["Plan"])
    iguales = [c for c in ["pg", "cuantico_s1", "clasico_s1", "reales"] if ej[(q, c)][0] == f]
    import statistics as st
    print(q, "igual a:", iguales, {c: round(st.median(ej[(q, c)][1]) / 1000, 1) for c in ["pg", "clasico_s1", "reales"]})
