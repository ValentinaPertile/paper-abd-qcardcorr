#!/usr/bin/env bash
# Corre el experimento completo de punta a punta. Se puede cortar y volver a lanzar: retoma
# desde la última etapa terminada (cada etapa deja un archivo .hecho en la carpeta etapas/).
#
# Antes de lanzarlo:   sudo service postgresql start
# Para lanzarlo:       nohup bash ~/paper_scripts/correr_todo.sh > ~/paper_scripts/progreso.log 2>&1 &
# Para ver el avance:  tail -n 30 ~/paper_scripts/progreso.log   (y el detalle de la etapa en curso: tail ~/paper_scripts/etapas/*.log)

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMDB="${IMDB:-$HOME/imdb}"
QCARD="${QCARD:-$HOME/QCardEst}"
LC="${LC:-$HOME/learnedcardinalities}"
DB="${DB:-imdb}"
REPS="${REPS:-5}"
SEMILLAS="${SEMILLAS:-1 2 3}"
EPISODIOS="${EPISODIOS:-1000}"
mkdir -p "$DIR/etapas"
paso() { echo; echo "===== $(date '+%d/%m %H:%M') | $* ====="; }
hecho() { [ -f "$DIR/etapas/$1.hecho" ]; }
marcar() { touch "$DIR/etapas/$1.hecho"; }
filas() { psql -d "$DB" -tAc "SELECT COUNT(*) FROM $1" 2>/dev/null || echo 0; }

# ---------- 0. Comprobaciones ----------
paso "0. Comprobaciones"
pg_isready -q || { echo "PostgreSQL no está corriendo. Ejecutá: sudo service postgresql start"; exit 1; }
[ -d "$IMDB" ] || { echo "No encuentro los CSV de IMDB en $IMDB"; exit 1; }
[ -d "$QCARD/qenv" ] || { echo "No encuentro el entorno de QCardEst en $QCARD/qenv"; exit 1; }
# shellcheck disable=SC1091
source "$QCARD/qenv/bin/activate"
python -c "import psycopg2, pandas, matplotlib" 2>/dev/null || {
  echo "Instalando psycopg2, pandas y matplotlib en el entorno de QCardEst..."
  if command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH" uv pip install psycopg2-binary pandas matplotlib
  else
    pip install psycopg2-binary pandas matplotlib
  fi
}
[ -d "$LC" ] || git clone --depth 1 https://github.com/andreaskipf/learnedcardinalities.git "$LC"
psql -d "$DB" -tAc "LOAD 'pg_hint_plan'" >/dev/null && echo "pg_hint_plan: OK"
free -h | sed -n '1,3p'

# ---------- 1. Carga de IMDB ----------
if ! hecho carga; then
  paso "1. Carga de IMDB (cada tabla se carga solo si está vacía)"
  cd "$IMDB"
  for csv in *.csv; do
    t="${csv%.csv}"
    lineas=$(wc -l < "$csv")
    if [ "$lineas" -gt 3000000 ]; then
      # tabla grande: se carga en partes y se anota cada parte terminada, para poder retomar
      if [ "$(filas "$t")" != "0" ] && ! ls "$DIR/etapas/parte_${t}_"* >/dev/null 2>&1; then
        echo "$t: ya estaba cargada ($(filas "$t") filas), se saltea"; continue
      fi
      echo "$t: $lineas líneas, se carga en partes de 2.000.000"
      rm -f "parte_${t}_"*
      split -l 2000000 -d -a 3 "$csv" "parte_${t}_"
      for p in parte_"${t}"_*; do
        if [ -f "$DIR/etapas/$p.hecho" ]; then rm -f "$p"; continue; fi
        psql -v ON_ERROR_STOP=1 -d "$DB" -c "\copy $t FROM '$p' WITH (FORMAT csv, ESCAPE '\')"
        touch "$DIR/etapas/$p.hecho"; rm -f "$p"
      done
    else
      n=$(filas "$t")
      if [ "$n" != "0" ]; then echo "$t: ya tiene $n filas, se saltea"; continue; fi
      psql -v ON_ERROR_STOP=1 -d "$DB" -c "\copy $t FROM '$csv' WITH (FORMAT csv, ESCAPE '\')"
    fi
    echo "$t: $(filas "$t") filas"
  done
  psql -v ON_ERROR_STOP=1 -d "$DB" -f "$DIR/00_indices.sql"
  ci=$(filas cast_info)
  if [ "$ci" != "36244344" ]; then echo "AVISO: cast_info tiene $ci filas (se esperaban 36244344)"; fi
  marcar carga
fi

# ---------- 2. Subplanes ----------
if ! hecho subplanes; then
  paso "2. Generación de subplanes"
  cd "$DIR"
  echo "Detalle en $DIR/etapas/subplanes.log"
  python -u 01_subplanes.py --workload "$LC/workloads/job-light.csv" --db "$DB" --salida datos > "$DIR/etapas/subplanes.log" 2>&1
  grep -E "AVISO|Listo" "$DIR/etapas/subplanes.log" || true
  n=$(wc -l < datos/subplanes.csv)
  [ "$n" = "440" ] || { echo "Se esperaban 440 subplanes y hay $n"; exit 1; }
  marcar subplanes
fi

# ---------- 3. Entrenamiento ----------
cd "$QCARD"
for m in "cuantico" "clasico" "cuantico --sin_separacion"; do
  et="entrenar_${m// /_}"
  if ! hecho "$et"; then
    paso "3. Entrenamiento: $m (semillas $SEMILLAS)"
    # shellcheck disable=SC2086
    echo "Detalle en $DIR/etapas/$et.log"
    PYTHONHASHSEED=0 python -u "$DIR/02_entrenar_cv.py" --datos "$DIR/datos" --modelo $m --semillas $SEMILLAS --episodios "$EPISODIOS" \
      > "$DIR/etapas/$et.log" 2>&1
    grep -E "^==|Listo" "$DIR/etapas/$et.log" || true
    marcar "$et"
  fi
done

# ---------- 4. Ejecución de consultas ----------
if ! hecho ejecucion; then
  paso "4. Ejecución de las 70 consultas con cada configuración ($REPS repeticiones)"
  cd "$DIR"
  echo "Detalle en $DIR/etapas/ejecucion.log"
  python -u 03_ejecutar.py --datos datos --db "$DB" --reps "$REPS" > "$DIR/etapas/ejecucion.log" 2>&1
  tail -n 3 "$DIR/etapas/ejecucion.log"
  marcar ejecucion
fi

# ---------- 5. Análisis ----------
paso "5. Análisis"
cd "$DIR"
python 04_analizar.py --datos datos
paso "TERMINADO. Resultados en $DIR/datos/resultados"
