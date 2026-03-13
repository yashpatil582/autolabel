"""Run the full AutoLabel pipeline and generate proof charts.

Usage:
    export GROQ_API_KEY=gsk_...
    python run_proof.py
"""
import os, sys, random, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    sys.exit("Error: GROQ_API_KEY not set. Add it to .env or export it.")

from autolabel.config import AutoLabelConfig
from autolabel.data.loaders import load_airline_tweets
from autolabel.evaluation.metrics import compute_f1
from autolabel.llm import get_provider
from autolabel.core.loop import AutonomousLoop

config = AutoLabelConfig()
dataset = load_airline_tweets(config.datasets_dir)
print(f"Dataset: {dataset.name} | {len(dataset.texts)} texts | {dataset.num_classes} classes")

# --- Classical baselines ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3), sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
pipeline.fit(dataset.train_texts, dataset.train_labels)
baseline_preds = pipeline.predict(dataset.test_texts).tolist()
baseline_f1 = compute_f1(dataset.test_labels, baseline_preds, dataset.label_space)
print(f"TF-IDF Baseline F1: {baseline_f1:.4f}")

random.seed(42)
random_preds = [random.choice(dataset.label_space) for _ in dataset.test_texts]
random_f1 = compute_f1(dataset.test_labels, random_preds, dataset.label_space)

majority_label = Counter(dataset.train_labels).most_common(1)[0][0]
majority_preds = [majority_label] * len(dataset.test_texts)
majority_f1 = compute_f1(dataset.test_labels, majority_preds, dataset.label_space)

# --- AutoLabel run ---
provider = get_provider('groq', model='llama-3.1-8b-instant')
loop = AutonomousLoop(
    dataset=dataset, provider=provider, config=config,
    label_model_type='majority', run_name='proof_v7_8b_mv_40iter',
)
results = loop.run(max_iterations=40)
test_result = loop.evaluate_test()
autolabel_f1 = test_result['f1']
print(f"\nAutoLabel Test F1: {autolabel_f1:.4f}")
print(f"Active LFs: {len(loop.registry.active_lfs)}")

# --- Chart 1: F1 trajectory ---
out_dir = config.experiments_dir
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
iterations = [r.iteration for r in results]
f1_values = [r.f1_after for r in results]
best_f1_traj = []
cur_best = 0.0
for r in results:
    if r.kept: cur_best = r.f1_after
    best_f1_traj.append(cur_best)

for it, f1, kept in zip(iterations, f1_values, [r.kept for r in results]):
    color = '#2ecc71' if kept else '#e74c3c'
    marker = 'o' if kept else 'x'
    ax.scatter(it, f1, c=color, marker=marker, s=80, zorder=3)

ax.plot(iterations, best_f1_traj, 'b-', linewidth=2.5, alpha=0.8, label='Best F1 (ratchet)')
ax.fill_between(iterations, 0, best_f1_traj, alpha=0.1, color='blue')
ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'TF-IDF Baseline ({baseline_f1:.2f})')
keep_patch = mpatches.Patch(color='#2ecc71', label='KEEP (F1 improved)')
discard_patch = mpatches.Patch(color='#e74c3c', label='DISCARD (no improvement)')
ax.legend(handles=[ax.get_lines()[0], ax.get_lines()[1], keep_patch, discard_patch], loc='lower right', fontsize=11)
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('F1 Score', fontsize=13)
ax.set_title('AutoLabel: Autonomous F1 Improvement Trajectory\n(Karpathy-style ratchet: keep if improved, discard otherwise)', fontsize=14)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(out_dir / 'f1_trajectory.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'f1_trajectory.png'}")

# --- Chart 2: Baseline comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Random', 'Majority\nClass', 'TF-IDF +\nLogReg', 'AutoLabel\n(Ours)']
f1_scores = [random_f1, majority_f1, baseline_f1, autolabel_f1]
colors = ['#95a5a6', '#95a5a6', '#e67e22', '#2ecc71']
bars = ax.bar(methods, f1_scores, color=colors, edgecolor='white', linewidth=1.5)
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score (micro)', fontsize=13)
ax.set_title('Airline Entity Extraction: AutoLabel vs Baselines', fontsize=14)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(out_dir / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'baseline_comparison.png'}")

# --- Chart 3: Strategy analysis ---
strategy_stats = {}
for r in results:
    s = r.strategy
    if s not in strategy_stats: strategy_stats[s] = {'tried': 0, 'kept': 0}
    strategy_stats[s]['tried'] += 1
    if r.kept: strategy_stats[s]['kept'] += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
strategies = sorted(strategy_stats.keys())
tried = [strategy_stats[s]['tried'] for s in strategies]
kept = [strategy_stats[s]['kept'] for s in strategies]
x = np.arange(len(strategies))
ax1.bar(x - 0.2, tried, 0.4, label='Tried', color='#3498db')
ax1.bar(x + 0.2, kept, 0.4, label='Kept', color='#2ecc71')
ax1.set_xticks(x); ax1.set_xticklabels(strategies, rotation=45, ha='right')
ax1.set_ylabel('Count'); ax1.set_title('Strategy Usage: Tried vs Kept'); ax1.legend()
success_rates = [strategy_stats[s]['kept'] / max(strategy_stats[s]['tried'], 1) for s in strategies]
ax2.bar(strategies, success_rates, color='#9b59b6')
ax2.set_ylabel('Keep Rate'); ax2.set_title('Strategy Success Rate'); ax2.set_ylim(0, 1.0)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(out_dir / 'strategy_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'strategy_analysis.png'}")

# --- Summary ---
print("\n" + "="*60)
print("AUTOLABEL: PROOF OF CONCEPT RESULTS")
print("="*60)
print(f"  Dataset:           Airline Tweets ({len(dataset.texts)} tweets, {dataset.num_classes} airlines)")
print(f"  Baseline (TF-IDF): {baseline_f1:.4f} F1")
print(f"  Random:            {random_f1:.4f} F1")
print(f"  Majority:          {majority_f1:.4f} F1")
print(f"  AutoLabel:         {autolabel_f1:.4f} F1")
print(f"  Improvement:       {(autolabel_f1 - baseline_f1):+.4f}")
print(f"  Iterations:        {len(results)}")
print(f"  Active LFs:        {len(loop.registry.active_lfs)}")
print(f"  Model:             llama-3.1-8b-instant (Groq)")
print("="*60)
