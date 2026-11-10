import csv, os, sys
import psycopg2
D = "/root/exp/datos"
meta = list(csv.DictReader(open(f"{D}/subplanes_meta.csv")))
comp = {m["consulta"]: m for m in meta if m["es_completa"] == "1"}
h = {}
for r in csv.DictReader(open("/root/sens/predicciones_cuantico_sin_separacion.csv")):
    h.setdefault(r["consulta"], []).append((r["alias"], int(r["filas_corregidas"])))
out = "/root/sens/tiempos_sens.csv"
hechas = {(r["consulta"], r["rep"]) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
f = open(out, "a"); w = csv.writer(f)
if not hechas and os.path.getsize(out) == 0: w.writerow(["consulta", "rep", "ms"])
con = psycopg2.connect(dbname="imdb"); con.autocommit = True; cur = con.cursor()
cur.execute("LOAD 'pg_hint_plan'"); cur.execute("SET max_parallel_workers_per_gather = 0"); cur.execute("SET statement_timeout = '10min'")
for q in ["65", "59"]:
    for rep in range(4):
        if (q, str(rep)) in hechas: continue
        cur.execute("/*+ " + " ".join(f"Rows({a} #{n})" for a, n in h[q]) + " */ EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) " + comp[q]["sql"])
        ms = cur.fetchone()[0][0]["Execution Time"]
        w.writerow([q, rep, f"{ms:.3f}"]); f.flush(); print(q, rep, ms, flush=True)
