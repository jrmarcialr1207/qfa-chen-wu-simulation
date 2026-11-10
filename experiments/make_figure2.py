import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/home/claude/results_table.json') as f:
    rows = json.load(f)

ns = [r[0] for r in rows]
k_gfa = [r[1] for r in rows]
n_pfa = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ax.plot(ns, k_gfa, marker='s', linewidth=2, markersize=7, label='Estados del GFA:  n^2  (Proposicion 3.1)')
ax.plot(ns, n_pfa, marker='o', linewidth=2, markersize=7, label='Estados totales del PFA:  2n^2 + 6  (Teorema 3.2)')

ax.set_xlabel('n  (estados del 1gQFA original)', fontsize=11)
ax.set_ylabel('Numero de estados', fontsize=11)
ax.set_title('Crecimiento cuadratico del costo de simulacion clasica', fontsize=12)
ax.set_xticks(ns)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9.5, loc='upper left')

plt.tight_layout()
plt.savefig('/home/claude/growth_figure.png', dpi=150)
print('Figura 2 guardada')
