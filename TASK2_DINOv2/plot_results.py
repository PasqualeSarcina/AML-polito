import matplotlib.pyplot as plt
import numpy as np

# --- DATA ENTRY ---

# Epochs (X-axis)
epochs = [1, 2, 3, 4, 5]

# EXPERIMENT 1: The "High LR" Run (LR=1e-4)
# (Inferred Epoch 1 from previous context, rest are from your logs)
exp1_train = [2.0200, 1.3478, 0.9675, 0.8298, 0.7583]
exp1_val   = [3.5500, 3.9255, 4.4000, 4.2102, 4.5407]

# EXPERIMENT 2: The "Low LR, 2 Layers" Run (LR=1e-5, n_layers=2)
exp2_train = [2.2527, 1.4950, 1.1974, 1.0127, 0.9006]
exp2_val   = [3.2790, 3.9180, 4.1811, 4.5789, 4.7892]

# EXPERIMENT 3: The "Best" Run (LR=1e-5, n_layers=1)
exp3_train = [2.8452, 2.1420, 1.8730, 1.7385, 1.6741]
exp3_val   = [3.1472, 3.2705, 3.4653, 3.5530, 3.6226]

# --- PLOTTING ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Training Loss Plot
ax1.plot(epochs, exp1_train, 'r--', marker='o', label='Exp 1: LR=1e-4 (High)')
ax1.plot(epochs, exp2_train, 'b--', marker='o', label='Exp 2: LR=1e-5 (2 Layers)')
ax1.plot(epochs, exp3_train, 'g-',  marker='o', linewidth=2, label='Exp 3: LR=1e-5 (1 Layer)')
ax1.set_title("Training Loss (Lower is usually better)", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2. Validation Loss Plot (The Important One)
ax2.plot(epochs, exp1_val, 'r--', marker='x', label='Exp 1: Overfitting Fast')
ax2.plot(epochs, exp2_val, 'b--', marker='x', label='Exp 2: Overfitting Slow')
ax2.plot(epochs, exp3_val, 'g-',  marker='o', linewidth=2, label='Exp 3: Best Model (Stable)')
ax2.set_title("Validation Loss (LOWER is BETTER)", fontsize=14)
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