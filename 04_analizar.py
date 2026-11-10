"""
Paso 4 del experimento: analizar los resultados y generar tablas y figuras para el paper.

Uso:
  python 04_analizar.py --datos datos
Salida en <datos>/resultados/:
  tabla1_qerror.csv        q-error por configuración (subplanes y consultas completas)
  tabla2_planes_tiempos.csv  planes que cambian y tiempos por configuración
  tabla3_por_tamano.csv    aceleración mediana según cantidad de tablas
  correlacion.txt          correlación entre mejora de q-error y mejora de tiempo
  figura1_qerror.png, figura2_aceleracion.png, figura3_correlacion.png  (blanco y negro)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def qerror(est, real):
    est = np.maximum(np.asarray(est, float), 1)
    real = np.maximum(np.asarray(real, float), 1)
    return np.maximum(est / real, real / est)


def resumen_q(q):
    return pd.Series({"mediana": np.median(q), "p90": np.percentile(q, 90), "p95": np.percentile(q, 95),
                      "maximo": np.max(q), "n": len(q)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    args = ap.parse_args()
    out = os.path.join(args.datos, "resultados")
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9})

    # ---------- q-error ----------
    meta = pd.read_csv(os.path.join(args.datos, "subplanes_meta.csv"))
    sub = pd.read_csv(os.path.join(args.datos, "subplanes.csv"), header=None,
                      names=["tablas", "sel", "costo_pg", "filas_pg", "t", "filas_reales"], skipinitialspace=True)
    base = meta.join(sub[["filas_pg", "filas_reales"]])
    filas = []
    for alcance, df in [("subplanes", base), ("consultas completas", base[base.es_completa == 1])]:
        filas.append(dict(config="pg", alcance=alcance, **resumen_q(qerror(df.filas_pg, df.filas_reales))))
    preds = []
    for nombre in ["cuantico", "clasico", "cuantico_sin_separacion", "clasico_sin_separacion"]:
        ruta = os.path.join(args.datos, f"predicciones_{nombre}.csv")
        if os.path.exists(ruta):
            p = pd.read_csv(ruta)
            p["nombre"] = nombre
            preds.append(p)
    for p in preds:
        for s, df in p.groupby("semilla"):
            for alcance, d in [("subplanes", df), ("consultas completas", df[df.es_completa == 1])]:
                filas.append(dict(config=f"{p.nombre.iloc[0]}_s{s}", alcance=alcance,
                                  **resumen_q(qerror(d.filas_corregidas, d.filas_reales))))
    t1 = pd.DataFrame(filas)
    t1.to_csv(os.path.join(out, "tabla1_qerror.csv"), index=False, float_format="%.2f")
    print("\nTabla 1 - q-error\n", t1.to_string(index=False, float_format="%.2f"))

    # figura 1: distribución del q-error en consultas completas
    grupos, etiquetas = [qerror(base[base.es_completa == 1].filas_pg, base[base.es_completa == 1].filas_reales)], ["PostgreSQL"]
    for p in preds:
        if "sin_separacion" in p.nombre.iloc[0]:
            continue
        d = p[p.es_completa == 1]
        grupos.append(qerror(d.filas_corregidas, d.filas_reales))
        etiquetas.append({"cuantico": "QCardCorr", "clasico": "Red clásica"}[p.nombre.iloc[0]])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.boxplot(grupos, showfliers=True, medianprops={"color": "black"},
               flierprops={"marker": ".", "markersize": 3})
    ax.set_xticks(range(1, len(etiquetas) + 1), etiquetas)
    ax.set_yscale("log"); ax.set_ylabel("q-error (escala log.)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figura1_qerror.png"), dpi=300); plt.close(fig)

    # ---------- planes y tiempos ----------
    ruta_ej = os.path.join(args.datos, "ejecuciones.csv")
    if not os.path.exists(ruta_ej):
        print("Todavía no hay ejecuciones.csv: se omite el análisis de tiempos")
        return
    ej = pd.read_csv(ruta_ej)
    ej = ej[(ej.rep > 0) & (ej.ms != "timeout")].copy()
    ej["ms"] = ej.ms.astype(float)
    med = ej.groupby(["consulta", "n_tablas", "config"]).agg(ms=("ms", "median"), firma=("firma_plan", "first"),
                                                           cv=("ms", lambda x: x.std() / x.mean())).reset_index()
    pg = med[med.config == "pg"].set_index("consulta")
    med["acel"] = med.apply(lambda r: pg.loc[r.consulta, "ms"] / r.ms, axis=1)
    med["plan_distinto"] = med.apply(lambda r: r.firma != pg.loc[r.consulta, "firma"], axis=1)
    t2 = med.groupby("config").agg(
        consultas=("consulta", "count"), planes_distintos_de_pg=("plan_distinto", "sum"),
        tiempo_total_s=("ms", lambda x: x.sum() / 1000), aceleracion_mediana=("acel", "median"),
        mas_rapidas_10pct=("acel", lambda x: (x > 1.1).sum()), mas_lentas_10pct=("acel", lambda x: (x < 1 / 1.1).sum()),
        variacion_media_repeticiones=("cv", "mean")).reset_index()
    t2.to_csv(os.path.join(out, "tabla2_planes_tiempos.csv"), index=False, float_format="%.3f")
    print("\nTabla 2 - planes y tiempos\n", t2.to_string(index=False, float_format="%.3f"))

    t3 = med.pivot_table(index="n_tablas", columns="config", values="acel", aggfunc="median")
    t3.to_csv(os.path.join(out, "tabla3_por_tamano.csv"), float_format="%.3f")
    print("\nTabla 3 - aceleración mediana por cantidad de tablas\n", t3.to_string(float_format="%.3f"))

    # figura 2: aceleración por consulta respecto de PostgreSQL
    cfgs = [c for c in sorted(med.config.unique()) if c != "pg"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.boxplot([med[med.config == c].acel for c in cfgs],
               medianprops={"color": "black"}, flierprops={"marker": ".", "markersize": 3})
    ax.set_xticks(range(1, len(cfgs) + 1), [c.replace("_", "\n") for c in cfgs])
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("Aceleración vs. PostgreSQL")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figura2_aceleracion.png"), dpi=300); plt.close(fig)

    # ---------- correlación entre mejora de estimación y mejora de tiempo ----------
    lineas = []
    puntos = []
    for p in preds:
        if "sin_separacion" in p.nombre.iloc[0]:
            continue
        for s, df in p.groupby("semilla"):
            cfg = f"{p.nombre.iloc[0]}_s{s}"
            d = df[df.es_completa == 1].copy()
            d["mejora_q"] = qerror(d.filas_pg, d.filas_reales) / qerror(d.filas_corregidas, d.filas_reales)
            d = d.merge(med[med.config == cfg][["consulta", "acel"]], on="consulta")
            if len(d) > 2:
                rho, pval = spearmanr(np.log(d.mejora_q), np.log(d.acel))
                lineas.append(f"{cfg}: rho de Spearman = {rho:.3f} (p = {pval:.3g}, n = {len(d)})")
                puntos.append(d)
    with open(os.path.join(out, "correlacion.txt"), "w") as f:
        f.write("Correlación entre la mejora del q-error de la consulta completa y la aceleración\n")
        f.write("\n".join(lineas) + "\n")
    print("\n" + "\n".join(lineas))
    if puntos:
        d = pd.concat(puntos)
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.scatter(d.mejora_q, d.acel, s=6, facecolors="none", edgecolors="black", linewidths=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(1, color="gray", lw=0.8, ls="--"); ax.axvline(1, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Mejora del q-error (veces)"); ax.set_ylabel("Aceleración (veces)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "figura3_correlacion.png"), dpi=300); plt.close(fig)
    print("\nResultados en", out)


if __name__ == "__main__":
    main()
