import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

with open('/home/claude/cn_table.json') as f:
    cn_data = json.load(f)

fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

L = np.arange(0, 9)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cn_data)))

for (n, k, N, C, CN), color in zip(cn_data, colors):
    log_signal = -L * np.log10(CN)
    ax.plot(L, log_signal, marker='o', markersize=4, linewidth=2,
            label=f'n={n}  (C*N approx {CN:,.0f})', color=color)

ax.axhline(-16, color='crimson', linestyle='--', linewidth=1.8,
           label='piso de precision float64 (approx 1e-16)')

ax.set_xlabel('Longitud de la palabra |w|', fontsize=11)
ax.set_ylabel('log10 de la magnitud relativa de la senal', fontsize=11)
ax.set_title('Decaimiento geometrico de la senal de cutpoint  ~ (C*N)^(-|w|)', fontsize=12)
ax.set_xlim(0, 8)
ax.set_ylim(-20, 1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig('/home/claude/decay_figure.png', dpi=150)
print('Figura guardada')
