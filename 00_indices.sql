-- Índices sobre las claves foráneas que usan los joins de JOB-light, y estadísticas actualizadas.
-- Con estos índices el optimizador tiene más planes posibles, y una mala estimación pesa más (Leis y otros, 2015).
CREATE INDEX IF NOT EXISTS mc_movie_id     ON movie_companies(movie_id);
CREATE INDEX IF NOT EXISTS mi_movie_id     ON movie_info(movie_id);
CREATE INDEX IF NOT EXISTS mi_idx_movie_id ON movie_info_idx(movie_id);
CREATE INDEX IF NOT EXISTS mk_movie_id     ON movie_keyword(movie_id);
CREATE INDEX IF NOT EXISTS ci_movie_id     ON cast_info(movie_id);
ANALYZE;
-- Comprobación: cantidad de filas de las tablas que usa JOB-light
SELECT 'title' AS tabla, COUNT(*) FROM title
UNION ALL SELECT 'cast_info', COUNT(*) FROM cast_info
UNION ALL SELECT 'movie_companies', COUNT(*) FROM movie_companies
UNION ALL SELECT 'movie_info', COUNT(*) FROM movie_info
UNION ALL SELECT 'movie_info_idx', COUNT(*) FROM movie_info_idx
UNION ALL SELECT 'movie_keyword', COUNT(*) FROM movie_keyword;
