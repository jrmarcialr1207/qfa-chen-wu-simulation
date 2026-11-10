# Implementación y experimentos: simulación probabilística exacta de autómatas cuánticos finitos

Este paquete contiene la implementación completa y los scripts de experimentación
usados para el informe "Simulación Probabilística Exacta de Autómatas Cuánticos
Finitos" (implementación de la construcción de Chen y Wu, arXiv:2604.07058).

## Requisitos

```
pip install numpy mpmath --break-system-packages
```

Todo el código usa solo `numpy` (álgebra lineal en punto flotante) y `mpmath`
(precisión arbitraria e intervalos, vía `mpmath.iv`). No se requieren otras
dependencias.

## Estructura

```
implementation/     -- Los cuatro módulos centrales (importables)
experiments/         -- Scripts que ejecutan los experimentos y generan las tablas
data/                -- Resultados ya calculados, en formato JSON (para no re-ejecutar todo)
figures/             -- Las figuras generadas (PNG)
```

### `implementation/qfa_simulation.py`
El módulo central. Implementa:
- La base ortonormal de operadores Hermitianos (Sección 5.2 del informe).
- Un 1gQFA aleatorio vía el método de dilatación de Stinespring (canales CPTP
  garantizados).
- La Proposición 3.1 de Chen y Wu (2026): linealización 1gQFA → GFA.
- El Teorema 3.2 de Chen y Wu (2026): construcción GFA → PFA, en tres variantes
  aritméticas: punto flotante (`gfa_to_pfa`), precisión arbitraria fija
  (`gfa_to_pfa_hp`, clase `PFA_HP`), y las clases `OneWayGeneralQFA`, `GFA`, `PFA`.
- `run_experiment(n, lam, ...)`: corre la batería completa de verificación
  para un valor de n y cutpoint dados, y devuelve todos los conteos de estados
  y discrepancias (usado para la Tabla 1).

### `implementation/qfa_improved.py`
Las dos Proposiciones de refinamiento (Sección 3 del informe):
- **Proposición 1** (cota ajustada): usa `tight_bound=True`.
- **Proposición 2** (normalización por símbolo): usa `per_symbol_C=True`.
- `gfa_to_pfa_improved(...)`: construcción refinada en punto flotante.
- `gfa_to_pfa_improved_hp(...)`: la misma construcción en precisión arbitraria.

### `implementation/qfa_interval.py`
La certificación por intervalos (Sección 4 del informe, Proposición 3):
- Base de Hermitianos, canales y GFA reconstruidos enteramente en
  `mpmath.iv` (aritmética de intervalos con redondeo dirigido).
- `gfa_to_pfa_iv(...)`: construcción del Teorema 3.2 (con los refinamientos)
  en intervalos.
- `certify_word(...)`: el procedimiento de duplicación adaptativa de precisión
  que certifica la decisión de cutpoint con una garantía matemática incondicional.

### `implementation/qfa_lower_bound.py`
La extensión a la cota inferior (Sección 2.4 / 6.6 del informe, Teorema 4.1
de Chen y Wu):
- Base ortonormal del subespacio de operadores Hermitianos de traza nula.
- Los estados "preparados" ρ_j y las mediciones M_s del argumento de shattering.
- `max_valid_epsilon(...)`: el margen ε máximo válido (positividad semidefinida).
- `certify_shattering(...)`: certificación por intervalos de la identidad
  Tr(M_s·ρ_j) = ε·s_j para todos los patrones de signo.

## Cómo reproducir cada tabla del informe

```bash
cd experiments/

# Tabla 1 (verificación de estados) y Tabla 2 (constantes C, N)
python3 generate_table1_table2.py

# Tabla 3 (comparación construcción original vs. refinada)
python3 generate_table3.py

# Tabla 4 (certificación por intervalos) -- tarda varios minutos
python3 run_interval_experiment.py

# Tabla 5 (certificación de la cota inferior) -- el caso n=4 exhaustivo
# (32768 patrones) toma ~90 segundos
python3 generate_table5.py

# Figuras 1 y 2 (requieren matplotlib: pip install matplotlib --break-system-packages)
python3 make_figure.py    # Figura 1: crecimiento de estados
python3 make_figure2.py   # Figura 2: decaimiento de la señal de cutpoint
```

Cada script imprime los resultados y los guarda como JSON en el directorio
donde se ejecute. Los archivos en `data/` son los resultados ya calculados
(exactamente los usados en el informe), por si se quiere inspeccionar los
números sin volver a correr nada.

## Notas de reproducibilidad

- Todos los 1gQFA aleatorios usan semillas fijas (`np.random.default_rng(100+n)`
  para las Tablas 1–3, `seed=7` o `seed=99` para las Tablas 4–5), por lo que
  volver a ejecutar los scripts debe reproducir exactamente los mismos números.
- `run_interval_experiment.py` y `generate_table5.py` son los más lentos
  (varios minutos en total), porque la aritmética de intervalos en Python puro
  (sin aceleración numérica de bajo nivel) es más costosa que `numpy`.
- El caso exhaustivo de n=4 en la Tabla 5 (32768 patrones de signo) es el
  paso individual más lento (~90 segundos).
