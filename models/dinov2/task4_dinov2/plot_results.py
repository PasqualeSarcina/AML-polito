import matplotlib.pyplot as plt
import numpy as np

# --- DATA ENTRY ---

# Epochs (X-axis)
epochs = [1, 2, 3, 4, 5]

# EXPERIMENT 1: First Run (LR=1e-4, Batch Size=5)
# Risultato: Overfitting rapido dopo ep 1.
exp1_train = [2.2234, 1.5996, 1.3826, 1.1763, 1.0211]
exp1_val   = [2.7441, 2.9191, 3.1801, 3.2766, 3.4100]

# EXPERIMENT 2: The "Safe Mode" Run (LR=1e-5, Batch Size=8)
# Risultato: Molto stabile ma lento (Underfitting). Best val alla fine.
exp2_train = [3.2551, 2.6684, 2.4777, 2.3809, 2.3396]
exp2_val   = [3.0835, 2.9691, 2.9341, 2.9259, 2.9391]

# EXPERIMENT 3: The "Aggressive + Batch 8" Run (LR=1e-4, Batch Size=8)
# Risultato: Simile a Exp 1, ottimo inizio ma poi Overfitting.
exp3_train = [2.3656, 1.7101, 1.4686, 1.3402, 1.2627]
exp3_val   = [2.8027, 2.9640, 3.1855, 3.3472, 3.3250]

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