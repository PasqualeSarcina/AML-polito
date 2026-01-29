import matplotlib.pyplot as plt
import numpy as np

# Epochs (X-axis)
epochs = [1, 2, 3, 4, 5]

# EXPERIMENT 1: (LR=1e-5, n_layers=1)
exp1_train = [3.2384, 2.3524, 1.9679, 1.7656, 1.6826]
exp1_val   = [2.9633, 2.8498, 2.8893, 2.9658, 2.9918]

# EXPERIMENT 2: Run (LR=1e-5, n_layers=2)
exp2_train = [2.5219, 1.3963, 0.9296, 0.7197, 0.6392]
exp2_val   = [2.7336, 3.3673, 3.7713, 4.0759, 4.1797]

# EXPERIMENT 3: (LR=1e-4, n_layers=1)
exp3_train = [1.5753, 0.4579, 0.1895, 0.1100, 0.0784]
exp3_val   = [3.8168, 4.3429, 4.7625, 5.1967, 5.3779]


# --- PLOTTING ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Training Loss Plot
ax1.plot(epochs, exp1_train, 'r--', marker='o', label='Exp 1: LR=1e-5 (1 Layer)')
ax1.plot(epochs, exp2_train, 'b--', marker='o', label='Exp 2: LR=1e-5 (2 Layers)')
ax1.plot(epochs, exp3_train, 'g-',  marker='o', linewidth=2, label='Exp 3: LR=1e-4 (1 Layer)')

ax1.set_title("Training Loss for Different Learning Rates and Fine-Tuning Depths", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2. Validation Loss Plot (The Important One)
ax2.plot(epochs, exp1_val, 'r--', marker='x',
         label='Exp 1: LR=1e-5, 1 Layer — Late overfit')

ax2.plot(epochs, exp2_val, 'b--', marker='x',
         label='Exp 2: LR=1e-5, 2 Layers — Early overfit')

ax2.plot(epochs, exp3_val, 'g-',  marker='o', linewidth=2,
         label='Exp 3: LR=1e-4, 1 Layer — Divergent')

ax2.set_title("Validation Loss and Generalization Behaviour", fontsize=14)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()


plt.tight_layout()
plt.show()

# Save the plot for your report
fig.savefig('training_comparison.png', dpi=300)
print("Plot saved as training_comparison.png")