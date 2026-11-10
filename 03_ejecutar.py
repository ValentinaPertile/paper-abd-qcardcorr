"""
Paso 3 del experimento: ejecutar las 70 consultas de JOB-light con cada configuración.

Configuraciones:
  pg              PostgreSQL sin modificar
  reales          se inyectan las cardinalidades reales de todos los subplanes (mejor caso posible)
  cuantico_sN     se inyectan las correcciones de QCardCorr (semilla N, validación cruzada)
  clasico_sN      se inyectan las correcciones de la red clásica (semilla N, validación cruzada)

Las cardinalidades se inyectan con pg_hint_plan: un hint Rows(alias... #filas) por subplan.
Cada consulta se ejecuta 1 vez de calentamiento (descartada) y --reps veces medidas, con
EXPLAIN (ANALYZE, FORMAT JSON). El orden de las configuraciones se mezcla en cada repetición.

Uso:
  python 03_ejecutar.py --datos datos --db imdb --reps 5
Salida: <datos>/ejecuciones.csv (se puede cortar y volver a correr: retoma)
"""
import argparse
import csv
import os
import random

import psycopg2


def firma(nodo):
    """Describe la forma del plan: tipo de cada operador y orden de las tablas."""
    hijos = nodo.get("Plans", [])
    tipo = nodo["Node Type"]
    if "Alias" in nodo:
        return f"{tipo}({nodo['Alias']})"
    if tipo in ("Aggregate", "Hash", "Materialize", "Sort", "Memoize", "Gather"):
        return firma(hijos[0]) if len(hijos) == 1 else tipo + "[" + ",".join(firma(h) for h in hijos) + "]"
    return tipo + "[" + ",".join(firma(h) for h in hijos) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--db", default="imdb")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout_min", type=int, default=10)
    ap.add_argument("--semillas", type=int, nargs="+", default=None,
                    help="semillas a usar (por defecto, todas las de los archivos de predicciones)")
    args = ap.parse_args()

    with open(os.path.join(args.datos, "subplanes_meta.csv")) as f:
        meta = list(csv.DictReader(f))
    reales = {}
    with open(os.path.join(args.datos, "subplanes.csv")) as f:
        for i, linea in enumerate(f):
            reales[str(i)] = int(linea.split(",")[-1])

    # cardinalidades a inyectar: config -> consulta -> lista de (alias, filas)
    hints = {"pg": {}, "reales": {}}
    for m in meta:
        hints["reales"].setdefault(m["consulta"], []).append((m["alias"], reales[m["fila"]]))
    for modelo in ("cuantico", "clasico"):
        ruta = os.path.join(args.datos, f"predicciones_{modelo}.csv")
        if not os.path.exists(ruta):
            print("No encuentro", ruta, "- se omite", modelo)
            continue
        with open(ruta) as f:
            for r in csv.DictReader(f):
                if args.semillas and int(r["semilla"]) not in args.semillas:
                    continue
                cfg = f"{modelo}_s{r['semilla']}"
                hints.setdefault(cfg, {}).setdefault(r["consulta"], []).append((r["alias"], int(r["filas_corregidas"])))

    completas = {m["consulta"]: m for m in meta if m["es_completa"] == "1"}

    ruta_out = os.path.join(args.datos, "ejecuciones.csv")
    hechas = set()
    if os.path.exists(ruta_out):
        with open(ruta_out) as f:
            for r in csv.DictReader(f):
                hechas.add((r["consulta"], r["config"], r["rep"]))
    nuevo = not os.path.exists(ruta_out)
    fout = open(ruta_out, "a", newline="")
    w = csv.writer(fout)
    if nuevo:
        w.writerow(["consulta", "n_tablas", "config", "rep", "ms", "costo_estimado", "filas_estimadas_raiz",
                    "filas_reales_raiz", "firma_plan"])

    con = psycopg2.connect(dbname=args.db)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("LOAD 'pg_hint_plan'")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute(f"SET statement_timeout = '{args.timeout_min}min'")

    rng = random.Random(0)
    configs = sorted(hints)
    for q in sorted(completas, key=int):
        sql = completas[q]["sql"]
        for rep in range(args.reps + 1):  # rep 0 = calentamiento
            orden = configs[:]
            rng.shuffle(orden)
            for cfg in orden:
                if (q, cfg, str(rep)) in hechas:
                    continue
                lista = hints[cfg].get(q, [])
                if cfg != "pg" and not lista:
                    continue
                bloque = "/*+ " + " ".join(f"Rows({a} #{n})" for a, n in lista) + " */ " if lista else ""
                try:
                    cur.execute(bloque + "EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) " + sql)
                    res = cur.fetchone()[0][0]
                    raiz = res["Plan"]["Plans"][0]
                    fila = [q, completas[q]["n_tablas"], cfg, rep, f"{res['Execution Time']:.3f}",
                            raiz["Total Cost"], raiz["Plan Rows"], raiz["Actual Rows"], firma(res["Plan"])]
                except psycopg2.errors.QueryCanceled:
                    fila = [q, completas[q]["n_tablas"], cfg, rep, "timeout", "", "", "", ""]
                w.writerow(fila)
                fout.flush()
                print(f"consulta {q:>2} rep {rep} {cfg:12s} {fila[4]:>10} ms  est={fila[6]} real={fila[7]}")
    print("Listo:", ruta_out)


if __name__ == "__main__":
    main()
