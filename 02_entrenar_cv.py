"""
Paso 2 del experimento: entrenar las correcciones con validación cruzada.

Divide las 70 consultas de JOB-light en 5 particiones estratificadas por cantidad
de tablas. Para cada partición entrena con los subplanes de las otras 4 y predice
los subplanes de la partición que quedó afuera. Así cada consulta se corrige con un
modelo que nunca la vio. Se repite con varias semillas.

Modelos:
  cuantico  QCardCorr (circuito variacional de QCardEst, simulado con Qiskit)
  clasico   red neuronal chica (grupo de control) con una cantidad de parámetros parecida

Con --sin_separacion entrena y evalúa con todas las consultas, como el trabajo original,
para medir cuánto cambian los resultados sin separar entrenamiento y prueba.

Se ejecuta desde la carpeta del repositorio QCardEst (usa su código):
  cd ~/QCardEst
  PYTHONHASHSEED=0 python ~/paper_scripts/02_entrenar_cv.py --datos ~/paper_scripts/datos --modelo cuantico --semillas 1 2 3
Salida: <datos>/predicciones_<modelo>[_sin_separacion].csv
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import cardEnv as cardEnvModule  # noqa: E402
from cardEnv import CardEnv  # noqa: E402
from ML import regressionInterpretation as interpretation  # noqa: E402
from ML.models import vqc  # noqa: E402
from ML.Regression import Regression  # noqa: E402


# QCardEst arma la lista de tablas con un set(), cuyo orden cambia entre ejecuciones de
# Python. Eso cambia qué número recibe cada tabla y por lo tanto los resultados.
# Se reemplaza por una lista ordenada para que el experimento sea reproducible.
def _tablas_ordenadas(file):
    if isinstance(file, str):
        file = open(file)
    tablas, maximo = set(), 0
    for linea in file:
        consulta = linea.split(",")[0].split(";")
        tablas.update(t.strip() for t in consulta)
        maximo = max(maximo, len(consulta))
    return {"nTables": len(tablas), "maxQuerySize": maximo, "tables": sorted(tablas)}


cardEnvModule.extractTablesFromCardsCSVfile = _tablas_ordenadas

AJUSTES = {
    "features": "simple", "encoding": ["rx", "rz"], "reuploading": False,
    "reps": 4, "calc": "yz", "entangleType": "circular", "entangle": "cx",
    "reward": "rational", "numEpisodes": 1000, "optimizer": "Adam",
    "lr": [0.01, 100, 0.9], "prefix": "CV", "noise": 0, "seed": 42,
    "batchsize": 1, "loss": "threshold", "data": "", "valueType": "rowFactor",
    "numFeatures": 6,
}


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s + 1)
    torch.manual_seed(s + 2)


def particiones(meta, k, semilla):
    """Asigna cada consulta a una partición, repartiendo parejo cada tamaño de consulta."""
    tam = {}
    for r in meta:
        tam[r["consulta"]] = max(tam.get(r["consulta"], 0), int(r["n_tablas"]))
    rng = random.Random(semilla)
    asignacion = {}
    for n in sorted(set(tam.values())):
        grupo = sorted(q for q, t in tam.items() if t == n)
        rng.shuffle(grupo)
        for i, q in enumerate(grupo):
            asignacion[q] = (i + len(asignacion)) % k
    return asignacion


class RedClasica(torch.nn.Module):
    def __init__(self, n_entradas, ocultas):
        super().__init__()
        self.red = torch.nn.Sequential(torch.nn.Linear(n_entradas, ocultas), torch.nn.Tanh(),
                                       torch.nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.red(x).squeeze(-1)


def entrenar_clasico(env, entrenamiento, episodios, ocultas, lr=0.01):
    n = len(entrenamiento[0]["features"])
    red = RedClasica(n, ocultas)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    for _ in range(episodios):
        e = random.choice(entrenamiento)
        pred = red(torch.tensor(e["features"], dtype=torch.float32))
        loss = (pred - env.expected(e)) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda e: red(torch.tensor(e["features"], dtype=torch.float32)).item(), sum(p.numel() for p in red.parameters())


def entrenar_cuantico(env, entrenamiento, ajustes):
    env.data = entrenamiento
    agente = interpretation.fromString(ajustes["loss"], None, env)
    modelo = vqc(settings=ajustes, nInputs=env.getInputSize(), nOutputs=agente.nInputs, norm=False)
    agente.setModel(modelo)
    Regression(agente, env, ajustes).run()

    def predecir(e):
        with torch.no_grad():
            return agente.predict(agente(torch.Tensor(e["features"]))).item()
    return predecir, sum(p.numel() for p in modelo.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpeta con subplanes.csv y subplanes_meta.csv")
    ap.add_argument("--modelo", choices=["cuantico", "clasico"], required=True)
    ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--episodios", type=int, default=1000)
    ap.add_argument("--capas", type=int, default=4)
    ap.add_argument("--ocultas", type=int, default=3)
    ap.add_argument("--sin_separacion", action="store_true")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("AVISO: corré el script con PYTHONHASHSEED=0 adelante para que sea reproducible")

    datos = os.path.abspath(args.datos)
    with open(os.path.join(datos, "subplanes_meta.csv")) as f:
        meta = list(csv.DictReader(f))
    for r in meta:
        r["consulta"] = int(r["consulta"])

    ajustes = dict(AJUSTES, numEpisodes=args.episodios, reps=args.capas)
    # CardEnv lee desde costs/<archivo>: se copia el CSV de subplanes ahí
    os.makedirs("costs/subplanes", exist_ok=True)
    with open(os.path.join(datos, "subplanes.csv")) as src, open("costs/subplanes/subplanes.csv", "w") as dst:
        dst.write(src.read())
    os.makedirs("results/settings", exist_ok=True)
    os.makedirs("results/solutions", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    sufijo = "_sin_separacion" if args.sin_separacion else ""
    ruta = os.path.join(datos, f"predicciones_{args.modelo}{sufijo}.csv")
    with open(ruta, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["modelo", "semilla", "particion", "fila", "consulta", "alias", "n_tablas", "es_completa",
                    "filas_pg", "filas_reales", "log_factor", "filas_corregidas", "parametros", "segundos"])
        for semilla in args.semillas:
            fijar_semilla(semilla)
            asign = particiones(meta, args.k, semilla)
            vueltas = [None] if args.sin_separacion else range(args.k)
            for p in vueltas:
                env = CardEnv(inputFile="subplanes/subplanes.csv", settings=dict(ajustes, seed=semilla,
                              prefix=f"CV_{args.modelo}_s{semilla}_p{p}"))
                todas = env.data
                if p is None:
                    entrenamiento, prueba = todas, todas
                else:
                    entrenamiento = [e for e, m in zip(todas, meta) if asign[m["consulta"]] != p]
                    prueba = [(e, m) for e, m in zip(todas, meta) if asign[m["consulta"]] == p]
                if p is None:
                    prueba = list(zip(todas, meta))
                t0 = time.time()
                fijar_semilla(semilla * 100 + (p or 0))
                if args.modelo == "cuantico":
                    predecir, nparam = entrenar_cuantico(env, entrenamiento, ajustes)
                else:
                    predecir, nparam = entrenar_clasico(env, entrenamiento, args.episodios, args.ocultas)
                seg = time.time() - t0
                for e, m in prueba:
                    lf = predecir(e)
                    w.writerow([args.modelo, semilla, "todas" if p is None else p, m["fila"], m["consulta"], m["alias"],
                                m["n_tablas"], m["es_completa"], e["rowsPredicted"], e["rows"], f"{lf:.6f}",
                                max(1, round(e["rowsPredicted"] * math.exp(lf))), nparam, f"{seg:.1f}"])
                fout.flush()
                print(f"== {args.modelo} semilla {semilla} partición {p}: {len(entrenamiento)} subplanes de entrenamiento, "
                      f"{len(prueba)} de prueba, {nparam} parámetros, {seg:.0f} s")
    print("Listo:", ruta)


if __name__ == "__main__":
    main()
