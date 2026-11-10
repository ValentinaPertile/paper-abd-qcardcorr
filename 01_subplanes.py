"""
Paso 1 del experimento: generar los subplanes de JOB-light.

Para cada una de las 70 consultas de JOB-light genera todas sus subconsultas
(title + cada subconjunto no vacío de las otras tablas) y obtiene:
  - la selectividad de los filtros de cada tabla,
  - la estimación de filas de PostgreSQL (EXPLAIN),
  - la cardinalidad real (SELECT COUNT(*)).

Salida (carpeta --salida):
  subplanes.csv       formato de QCardEst: tablas;... , selectividades;... , costo_pg, filas_pg, 0, filas_reales
  subplanes_meta.csv  una fila por subplan: fila, consulta, alias, n_tablas, es_completa, sql
Se puede cortar y volver a correr: retoma desde lo que ya calculó.

Uso:
  python 01_subplanes.py --workload ~/learnedcardinalities/workloads/job-light.csv --db imdb --salida datos
"""
import argparse
import csv
import itertools
import json
import os
import time

import psycopg2

ALIAS_A_TABLA = {}  # se completa al leer el workload


def leer_workload(ruta):
    consultas = []
    with open(ruta) as f:
        for i, linea in enumerate(f):
            tablas, joins, preds, card = linea.strip().split("#")
            alias = {}
            for t in tablas.split(","):
                nombre, a = t.split()
                alias[a] = nombre
                ALIAS_A_TABLA[a] = nombre
            p = preds.split(",") if preds else []
            filtros = {}
            for k in range(0, len(p), 3):
                col, op, val = p[k], p[k + 1], p[k + 2]
                a = col.split(".")[0]
                filtros.setdefault(a, []).append(f"{col}{op}{val}")
            consultas.append({"id": i, "alias": alias, "filtros": filtros, "card_workload": int(card)})
    return consultas


def sql_subplan(alias_sub, consulta):
    desde = ",".join(f"{consulta['alias'][a]} {a}" for a in sorted(alias_sub))
    cond = [f"t.id={a}.movie_id" for a in sorted(alias_sub) if a != "t"]
    for a in sorted(alias_sub):
        cond += consulta["filtros"].get(a, [])
    where = " AND ".join(cond) if cond else "TRUE"
    return f"SELECT COUNT(*) FROM {desde} WHERE {where}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--db", default="imdb")
    ap.add_argument("--salida", default="datos")
    ap.add_argument("--timeout_min", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    consultas = leer_workload(args.workload)
    con = psycopg2.connect(dbname=args.db)
    con.autocommit = True
    cur = con.cursor()
    cur.execute(f"SET statement_timeout = '{args.timeout_min}min'")

    # cantidad de filas de cada tabla y selectividad de cada combinación tabla+filtros
    total = {}
    selectividad = {}

    def sel(a, consulta):
        tabla = consulta["alias"][a]
        filtros = consulta["filtros"].get(a, [])
        if not filtros:
            return 1.0
        clave = (tabla, tuple(sorted(filtros)))
        if clave not in selectividad:
            if tabla not in total:
                cur.execute(f"SELECT COUNT(*) FROM {tabla}")
                total[tabla] = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {tabla} {a} WHERE " + " AND ".join(filtros))
            selectividad[clave] = cur.fetchone()[0] / total[tabla]
        return selectividad[clave]

    ruta_meta = os.path.join(args.salida, "subplanes_meta.csv")
    ruta_datos = os.path.join(args.salida, "subplanes.csv")
    hechos = set()
    if os.path.exists(ruta_meta):
        with open(ruta_meta) as f:
            for r in csv.DictReader(f):
                hechos.add((int(r["consulta"]), r["alias"]))
    nueva = not os.path.exists(ruta_meta)
    fmeta = open(ruta_meta, "a", newline="")
    fdatos = open(ruta_datos, "a")
    wmeta = csv.writer(fmeta)
    if nueva:
        wmeta.writerow(["fila", "consulta", "alias", "n_tablas", "es_completa", "sql"])
    fila = len(hechos)

    for c in consultas:
        otros = sorted(a for a in c["alias"] if a != "t")
        for k in range(1, len(otros) + 1):
            for comb in itertools.combinations(otros, k):
                sub = ("t",) + comb
                clave_alias = " ".join(sorted(sub))
                if (c["id"], clave_alias) in hechos:
                    continue
                sql = sql_subplan(sub, c)
                # estimación de PostgreSQL, sin paralelismo para que el plan sea simple
                cur.execute("SET max_parallel_workers_per_gather = 0")
                cur.execute("EXPLAIN (FORMAT JSON) " + sql)
                plan = cur.fetchone()[0][0]["Plan"]
                nodo = plan["Plans"][0]  # hijo del Aggregate: el join completo de este subplan
                filas_pg = max(1, int(nodo["Plan Rows"]))
                costo_pg = nodo["Total Cost"]
                # cardinalidad real (se permite paralelismo para que termine antes)
                cur.execute("SET max_parallel_workers_per_gather = 2")
                t0 = time.time()
                cur.execute(sql)
                filas_reales = cur.fetchone()[0]
                seg = time.time() - t0
                # QCardEst trabaja con logaritmos: una cardinalidad 0 se guarda como 1
                filas_reales_log = max(1, filas_reales)
                orden = sorted(sub, key=lambda a: c["alias"][a])
                tablas = ";".join(c["alias"][a] for a in orden)
                sels = ";".join(repr(sel(a, c)) for a in orden)
                fdatos.write(f"{tablas},{sels}, {costo_pg} , {filas_pg} , 0 , {filas_reales_log}\n")
                es_completa = int(len(sub) == len(c["alias"]))
                wmeta.writerow([fila, c["id"], clave_alias, len(sub), es_completa, sql])
                fdatos.flush(); fmeta.flush()
                print(f"consulta {c['id']:2d}  {clave_alias:22s} pg={filas_pg:>10d}  real={filas_reales:>10d}  ({seg:.1f} s)")
                if es_completa and filas_reales != c["card_workload"]:
                    print(f"   AVISO: la cardinalidad real ({filas_reales}) no coincide con la del workload ({c['card_workload']})")
                fila += 1
    print(f"Listo: {fila} subplanes en {ruta_datos}")


if __name__ == "__main__":
    main()
