"""Resume los resultados del experimento en un JSON y genera las figuras para el paper."""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

D = sys.argv[1]
OUT = sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8})

meta = list(csv.DictReader(open(f"{D}/subplanes_meta.csv")))
sub = [l.strip().split(",") for l in open(f"{D}/subplanes.csv")]
for m in meta:
    f = sub[int(m["fila"])]
    m["pg"], m["real"] = float(f[3]), float(f[5])
comp = {m["consulta"]: m for m in meta if m["es_completa"] == "1"}


def q(e, r):
    e, r = np.maximum(np.asarray(e, float), 1), np.maximum(np.asarray(r, float), 1)
    return np.maximum(e / r, r / e)


def stats(v):
    return {"mediana": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


R = {"qerror": {}, "por_semilla": {}}
R["qerror"]["PostgreSQL"] = {"subplanes": stats(q([m["pg"] for m in meta], [m["real"] for m in meta])),
                             "completas": stats(q([m["pg"] for m in comp.values()], [m["real"] for m in comp.values()]))}
preds = {}
for nombre in ["cuantico", "clasico", "cuantico_sin_separacion"]:
    filas = list(csv.DictReader(open(f"{D}/predicciones_{nombre}.csv")))
    preds[nombre] = filas
    por = {}
    for s in sorted({r["semilla"] for r in filas}):
        rr = [r for r in filas if r["semilla"] == s]
        cc = [r for r in rr if r["es_completa"] == "1"]
        qs = q([float(r["filas_corregidas"]) for r in rr], [float(r["filas_reales"]) for r in rr])
        qp = q([float(r["filas_pg"]) for r in rr], [float(r["filas_reales"]) for r in rr])
        por[s] = {"subplanes": stats(qs), "completas": stats(q([float(r["filas_corregidas"]) for r in cc], [float(r["filas_reales"]) for r in cc])),
                  "mejoran": int((qs < qp).sum()), "empeoran": int((qs > qp).sum()), "n": len(rr),
                  "lf_max": float(max(abs(float(r["log_factor"])) for r in rr))}
    R["por_semilla"][nombre] = por
    R["qerror"][nombre] = {a: {k: float(np.mean([por[s][a][k] for s in por])) for k in ["mediana", "p95", "max"]}
                           for a in ["subplanes", "completas"]}
    R["qerror"][nombre]["mejoran_media"] = float(np.mean([por[s]["mejoran"] for s in por]))
    R["qerror"][nombre]["empeoran_media"] = float(np.mean([por[s]["empeoran"] for s in por]))

# tiempos y planes
ej_todas = list(csv.DictReader(open(f"{D}/ejecuciones.csv")))
R["timeouts"] = sorted({(r["consulta"], r["config"]) for r in ej_todas if r["ms"] == "timeout"})
ej = [r for r in ej_todas if r["rep"] != "0" and r["ms"] != "timeout"]
med, firma, ntab = {}, {}, {}
for r in ej:
    med.setdefault((r["consulta"], r["config"]), []).append(float(r["ms"]))
    firma[(r["consulta"], r["config"])] = r["firma_plan"]
    ntab[r["consulta"]] = int(r["n_tablas"])
cfgs = ["pg", "cuantico_s1", "clasico_s1", "reales"]
consultas = sorted({k[0] for k in med if all((k[0], c) in med for c in cfgs)}, key=int)
T = {}
acel = {}
for c in cfgs:
    ms = {qq: np.median(med[(qq, c)]) for qq in consultas}
    a = np.array([np.median(med[(qq, "pg")]) / ms[qq] for qq in consultas])
    acel[c] = a
    T[c] = {"planes_distintos": int(sum(firma[(qq, c)] != firma[(qq, "pg")] for qq in consultas)),
            "acel_mediana": float(np.median(a)), "tiempo_total_s": float(sum(ms.values()) / 1000),
            "mas_rapidas": int((a > 1.1).sum()), "mas_lentas": int((a < 1 / 1.1).sum()),
            "cv_medio": float(np.mean([np.std(med[(qq, c)]) / np.mean(med[(qq, c)]) for qq in consultas]))}
    T[c]["por_tablas"] = {str(n): float(np.median([a[i] for i, qq in enumerate(consultas) if ntab[qq] == n]))
                          for n in sorted(set(ntab.values()))}
qmax = max(consultas, key=lambda qq: np.median(med[(qq, "pg")]))
for c in cfgs:
    T[c]["tiempo_sin_mayor_s"] = float(sum(np.median(med[(qq, c)]) for qq in consultas if qq != qmax) / 1000)
    T[c]["cambios"] = [{"consulta": qq, "tablas": ntab[qq], "pg_s": float(np.median(med[(qq, "pg")]) / 1000),
                        "cfg_s": float(np.median(med[(qq, c)]) / 1000), "acel": float(acel[c][i]),
                        "q_pg": float(q(comp[qq]["pg"], comp[qq]["real"]))}
                       for i, qq in enumerate(consultas) if firma[(qq, c)] != firma[(qq, "pg")]]
R["consulta_mayor"] = {"consulta": qmax, "tablas": ntab[qmax], "pg_s": float(np.median(med[(qmax, "pg")]) / 1000)}
R["tiempos"] = T
R["n_consultas"] = len(consultas)
R["consultas_por_tablas"] = {str(n): sum(1 for qq in consultas if ntab[qq] == n) for n in sorted(set(ntab.values()))}
# consultas donde las cardinalidades reales cambian el plan
R["reales_cambios"] = [{"consulta": qq, "tablas": ntab[qq], "acel": float(acel["reales"][i]),
                        "q_pg": float(q(comp[qq]["pg"], comp[qq]["real"]))}
                       for i, qq in enumerate(consultas) if firma[(qq, "reales")] != firma[(qq, "pg")]]
# correlación mejora de q-error vs aceleración
corr = {}
for nombre, cfg in [("cuantico", "cuantico_s1"), ("clasico", "clasico_s1")]:
    pr = {r["consulta"]: r for r in preds[nombre] if r["semilla"] == "1" and r["es_completa"] == "1"}
    mej = [float(q(float(pr[qq]["filas_pg"]), float(pr[qq]["filas_reales"])) / q(float(pr[qq]["filas_corregidas"]), float(pr[qq]["filas_reales"])))
           for qq in consultas]
    rho, p = spearmanr(np.log(mej), np.log(acel[cfg]))
    corr[nombre] = {"rho": float(rho), "p": float(p)}
R["correlacion"] = corr
json.dump(R, open(f"{OUT}/resultados.json", "w"), indent=1)

# figura 1: q-error de las consultas completas
grupos = [q([m["pg"] for m in comp.values()], [m["real"] for m in comp.values()])]
for nombre in ["cuantico", "clasico", "cuantico_sin_separacion"]:
    cc = [r for r in preds[nombre] if r["semilla"] == "1" and r["es_completa"] == "1"]
    grupos.append(q([float(r["filas_corregidas"]) for r in cc], [float(r["filas_reales"]) for r in cc]))
fig, ax = plt.subplots(figsize=(3.15, 2.1))
ax.boxplot(grupos, widths=0.5, medianprops={"color": "black"}, flierprops={"marker": ".", "markersize": 3})
ax.set_xticks([1, 2, 3, 4], ["PostgreSQL", "QCardCorr", "Red clásica", "QCardCorr\nsin separación"])
ax.set_yscale("log"); ax.set_ylabel("q-error (escala logarítmica)")
fig.tight_layout(); fig.savefig(f"{OUT}/figura1.png", dpi=300); plt.close(fig)

# figura 2: aceleración respecto de PostgreSQL según cantidad de tablas
fig, ax = plt.subplots(figsize=(3.15, 2.1))
marcas = {"cuantico_s1": ("o", "QCardCorr"), "clasico_s1": ("s", "Red clásica"), "reales": ("^", "Cardinalidades reales")}
desp = {"cuantico_s1": -0.18, "clasico_s1": 0.0, "reales": 0.18}
for c, (mk, lab) in marcas.items():
    x = [ntab[qq] + desp[c] for qq in consultas]
    ax.scatter(x, acel[c], marker=mk, s=10, facecolors="none", edgecolors="black", linewidths=0.6, label=lab)
ax.axhline(1, color="gray", lw=0.7, ls="--")
ax.set_yscale("log"); ax.set_xticks(sorted(set(ntab.values())))
ax.set_xlabel("Tablas por consulta"); ax.set_ylabel("Aceleración vs. PostgreSQL")
ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); fig.savefig(f"{OUT}/figura2.png", dpi=300); plt.close(fig)
print(json.dumps({k: R[k] for k in ["n_consultas", "tiempos", "correlacion"]}, indent=1)[:2500])
