import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 300
})

N_subjects = 12
art_base = [2, 3, 2, 2, 4, 3, 2, 2, 3, 2, 3, 2] 
art_npps = [4, 5, 4, 4, 5, 4, 5, 4, 4, 5, 4, 5]
real_base = [3, 4, 3, 3, 4, 3, 4, 3, 3, 4, 3, 4]
real_npps = [4, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4, 4]
pref_base = [2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2]
pref_npps = [4, 5, 4, 4, 4, 5, 4, 4, 5, 4, 4, 5]

data = [
    [art_base, art_npps],
    [real_base, real_npps],
    [pref_base, pref_npps]
]

labels_metrics = ['Absence of\nArtifacts', 'Acoustic\nRealism', 'Overall\nPreference']
labels_systems = ['Baseline (FluidSynth)', 'NPPS (Ours)']
colors = ['#7d94b5', '#d62728'] 

def calc_stats(scores):
    mean = np.mean(scores)
    ci = stats.t.ppf(1 - 0.05 / 2, len(scores) - 1) * stats.sem(scores)
    return mean, ci

means = np.zeros((3, 2))
cis = np.zeros((3, 2))

for m in range(3):
    for s in range(2):
        means[m, s], cis[m, s] = calc_stats(data[m][s])

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels_metrics))
width = 0.35

rects1 = ax.bar(x - width/2, means[:, 0], width, yerr=cis[:, 0], 
                label=labels_systems[0], color=colors[0], capsize=6, edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, means[:, 1], width, yerr=cis[:, 1], 
                label=labels_systems[1], color=colors[1], capsize=6, edgecolor='black', alpha=0.9)

ax.set_ylabel('Mean Opinion Score (1-5)', fontweight='bold')

ax.set_title('Subjective Perceptual Evaluation (with 95% CI)', fontweight='bold', pad=45)
ax.set_xticks(x)
ax.set_xticklabels(labels_metrics, fontweight='bold')

# Raise the upper limit of the Y-axis to prevent data or error bars from hitting the top
ax.set_ylim(1.0, 6.2) 

# The legend is placed at the top edge of the plotting area, precisely between the title and the bar chart
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, edgecolor='black')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 20),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig_mos_evaluation.pdf", bbox_inches='tight')
plt.show()