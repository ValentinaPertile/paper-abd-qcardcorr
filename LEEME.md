# Experimento del paper: guía para correrlo en WSL

Todo se corre en la terminal de **Ubuntu** (la línea termina en `$`, no empieza con `C:\`).

## Forma rápida: todo de una vez

```bash
sudo service postgresql start
nohup bash ~/paper_scripts/correr_todo.sh > ~/paper_scripts/progreso.log 2>&1 &
```

Queda corriendo solo (puede tardar entre 10 y 15 horas). Para ver en qué etapa va:

```bash
tail -n 30 ~/paper_scripts/progreso.log
```

Si se corta (por ejemplo, si se apaga la PC o se cierra WSL), volvé a correr los dos comandos: sigue desde donde quedó.
Cuando termina, `progreso.log` dice `TERMINADO` y los resultados quedan en `~/paper_scripts/datos/resultados/`.

Abajo está el detalle de cada etapa, por si hay que correr alguna a mano.

## 0. Preparar (una sola vez)

```bash
# copiar esta carpeta a tu Ubuntu, por ejemplo en ~/paper_scripts
cd ~/QCardEst
source qenv/bin/activate
uv pip install psycopg2-binary pandas matplotlib
cd ~
git clone https://github.com/andreaskipf/learnedcardinalities.git   # tiene las 70 consultas de JOB-light
sudo service postgresql start
```

## 1. Terminar de cargar IMDB

Ver el mensaje del chat (carga de `cast_info` por partes). Después:

```bash
psql -d imdb -f ~/paper_scripts/00_indices.sql
```

## 2. Generar los subplanes (30 a 90 minutos)

```bash
cd ~/paper_scripts
python 01_subplanes.py --workload ~/learnedcardinalities/workloads/job-light.csv --db imdb --salida datos
```

Tiene que terminar con `Listo: 440 subplanes`. No debería aparecer ningún `AVISO`
(si aparece, la carga de IMDB quedó incompleta). Si se corta, volvé a correr el mismo comando: sigue desde donde quedó.

## 3. Entrenar las correcciones (unas 3 a 4 horas: conviene dejarlo corriendo)

```bash
cd ~/QCardEst
PYTHONHASHSEED=0 python ~/paper_scripts/02_entrenar_cv.py --datos ~/paper_scripts/datos --modelo cuantico --semillas 1 2 3
PYTHONHASHSEED=0 python ~/paper_scripts/02_entrenar_cv.py --datos ~/paper_scripts/datos --modelo clasico  --semillas 1 2 3
# comparación con el método original (sin separar entrenamiento y prueba):
PYTHONHASHSEED=0 python ~/paper_scripts/02_entrenar_cv.py --datos ~/paper_scripts/datos --modelo cuantico --semillas 1 2 3 --sin_separacion
```

## 4. Ejecutar las consultas (varias horas: dejarlo corriendo de noche)

```bash
cd ~/paper_scripts
python 03_ejecutar.py --datos datos --db imdb --reps 5
```

Si se corta, volvé a correr el mismo comando: sigue desde donde quedó.

## 5. Analizar

```bash
python 04_analizar.py --datos datos
```

Las tablas y figuras quedan en `datos/resultados/`. Mandá esa carpeta al chat.

## Qué hace cada script

| Script | Qué hace | Sección del paper |
|---|---|---|
| `01_subplanes.py` | Genera los 440 subplanes de JOB-light con su estimación de PostgreSQL y su cardinalidad real | Diseño experimental: datos |
| `02_entrenar_cv.py` | Entrena QCardCorr y la red clásica con validación cruzada de 5 particiones | Diseño experimental: configuraciones |
| `03_ejecutar.py` | Ejecuta las 70 consultas inyectando cada conjunto de cardinalidades con pg_hint_plan | Diseño experimental: medición |
| `04_analizar.py` | q-error, planes que cambian, tiempos, correlación; tablas y figuras en blanco y negro | Resultados |
