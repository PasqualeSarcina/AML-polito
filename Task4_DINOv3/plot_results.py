import matplotlib.pyplot as plt
import numpy as np

# --- DATA ENTRY ---

# Epochs (X-axis)
epochs = [1, 2, 3]

# EXPERIMENT 1: First Run (LR=1e-4 Batch Size=8)
exp1_train = [2.6911, 2.0850, 1.8327]
exp1_val   = [2.6616, 2.6300, 2.7220]


# EXPERIMENT 2: (LR=1e-5 Batch Size=8)
exp2_train = [3.5391, 2.9895, 2.8719]
exp2_val   = [3.1845, 3.0378, 3.0072]

# EXPERIMENT 3: (LR=1e-5 Batch Size=4)
exp3_train = [3.3736, 2.8653, 2.7487]
exp3_val   = [3.0912, 2.9674, 2.9280]

# --- PLOTTING ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Training Loss Plot
ax1.plot(epochs, exp1_train, 'r--', marker='o', label='Exp 1: LR=1e-4')
ax1.plot(epochs, exp2_train, 'b--', marker='o', label='Exp 2: LR=1e-5 ')
ax1.plot(epochs, exp3_train, 'g-',  marker='o', linewidth=2, label='Exp 3: LR=1e-5')
ax1.set_title("Training Loss (Lower is usually better)", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2. Validation Loss Plot (The Important One)
ax2.plot(epochs, exp2_val, 'r--', marker='x', label='Exp 1: Stable')
ax2.plot(epochs, exp3_val, 'b--', marker='x', label='Exp 2: Overfitting fast')
ax2.plot(epochs, exp1_val, 'g-',  marker='o', linewidth=2, label='Best model')
ax2.set_title("Validation Loss (LOWER is BETTER)", fontsize=14)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Highlight the best point
best_loss = min(exp1_val)
best_epoch = exp1_val.index(best_loss) + 1
ax2.annotate(f'Best Model: {best_loss:.4f}', 
             xy=(best_epoch, best_loss), 
             xytext=(best_epoch, best_loss - 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()

# Save the plot for your report
fig.savefig('training_comparison.png', dpi=300)
print("Plot saved as training_comparison.png")