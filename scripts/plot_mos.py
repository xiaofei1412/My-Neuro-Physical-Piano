import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Rigorous academic typesetting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

# 1. Experimental data input area
# Scoring Criteria (1-5): 1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent
# Matrix shape: [evaluation dimensions, systems, subjects]

# There are 12 subjects in the experiment
N_subjects = 12

# Dimension 1: Absence of Artifacts
art_base = [2, 3, 2, 2, 4, 3, 2, 2, 3, 2, 3, 2] 
art_npps = [4, 5, 4, 4, 5, 4, 5, 4, 4, 5, 4, 5]

# Dimension 2: Acoustic Realism
real_base = [3, 4, 3, 3, 4, 3, 4, 3, 3, 4, 3, 4]
real_npps = [4, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4, 4]

# Dimension 3: Overall Preference
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

# 2. Statistical calculations (mean and 95% confidence interval)
def calc_stats(scores):
    mean = np.mean(scores)
    # Calculate the error bounds of the 95% confidence interval
    ci = stats.t.ppf(1 - 0.05 / 2, len(scores) - 1) * stats.sem(scores)
    return mean, ci

means = np.zeros((3, 2))
cis = np.zeros((3, 2))

for m in range(3):
    for s in range(2):
        means[m, s], cis[m, s] = calc_stats(data[m][s])

# 3. Draw an academic bar chart with an error bar
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(labels_metrics))
width = 0.35

# Draw columns and error bars
rects1 = ax.bar(x - width/2, means[:, 0], width, yerr=cis[:, 0], 
                label=labels_systems[0], color=colors[0], capsize=5, edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, means[:, 1], width, yerr=cis[:, 1], 
                label=labels_systems[1], color=colors[1], capsize=5, edgecolor='black', alpha=0.9)

ax.set_ylabel('Mean Opinion Score (1-5)', fontweight='bold')
ax.set_title('Subjective Perceptual Evaluation (with 95% CI)', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels_metrics, fontweight='bold')
ax.set_ylim(1, 5.5) # MOS scores are typically labeled 1-5, leaving space for a legend

# Compact Legend
ax.legend(loc='upper left', frameon=True, edgecolor='black')

# Mark the specific values ​​on the pillar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 15),  # Shift upwards to avoid error bars
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig_mos_evaluation.pdf", bbox_inches='tight')
plt.show()

# Terminal output statistical summary for paper insertion
print("MOS Statistical Report:")
for m_idx, metric in enumerate(labels_metrics):
    print(f"[{metric.replace(chr(10), ' ')}]")
    for s_idx, sys_name in enumerate(labels_systems):
        print(f"  - {sys_name}: {means[m_idx, s_idx]:.2f} ± {cis[m_idx, s_idx]:.2f}")