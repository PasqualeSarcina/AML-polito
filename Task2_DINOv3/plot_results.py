import matplotlib.pyplot as plt
import numpy as np

# Epochs (X-axis)
epochs = [1, 2, 3, 4, 5]

# EXPERIMENT 1: (LR=1e-5, n_layers=1)
exp1_train = [3.3955, 2.7276, 2.4855, 2.3630, 2.3099]
exp1_val   = [3.0707, 2.8727, 2.8012, 2.7776, 2.7671]

# EXPERIMENT 2: Run (LR=1e-5, n_layers=2)
exp2_train = [2.8348, 2.1053, 1.8428, 1.7101, 1.4868]
exp2_val   = [2.6709, 2.6489, 2.6867, 2.7942, 3.0346]

# EXPERIMENT 3: (LR=1e-4, n_layers=1)
exp3_train = [2.2723, 1.5257, 1.2649, 1.1229, 1.0356]
exp3_val   = [2.8253, 2.9629, 3.2489, 3.4966, 3.6891]

# --- PLOTTING ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Training Loss Plot
ax1.plot(epochs, exp1_train, 'r--', marker='o', label='Exp 1: LR=1e-4 (1 Layer)')
ax1.plot(epochs, exp2_train, 'b--', marker='o', label='Exp 2: LR=1e-5 (2 Layers)')
ax1.plot(epochs, exp3_train, 'g-',  marker='o', linewidth=2, label='Exp 3: LR=1e-5 (1 Layer)')
ax1.set_title("--)", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2. Validation Loss Plot (The Important One)
ax2.plot(epochs, exp1_val, 'r--', marker='x', label='Exp 1: Stable')
ax2.plot(epochs, exp2_val, 'b--', marker='x', label='Exp 2: Overfitting Slow')
ax2.plot(epochs, exp3_val, 'g-',  marker='o', linewidth=2, label='Exp 3: Best Model (Stable)')
ax2.set_title("---", fontsize=14)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Highlight the best point
best_loss = min(exp3_val)
best_epoch = exp3_val.index(best_loss) + 1
ax2.annotate(f'Best Model: {best_loss:.4f}', 
             xy=(best_epoch, best_loss), 
             xytext=(best_epoch, best_loss - 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()

# Save the plot for your report
fig.savefig('training_comparison.png', dpi=300)
print("Plot saved as training_comparison.png")