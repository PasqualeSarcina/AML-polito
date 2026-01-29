import matplotlib.pyplot as plt

epochs = [1, 2, 3]

# EXPERIMENT 1: First Run (LR=1e-4 Batch Size=8)
exp1_train = [2.2945, 1.3929, 1.0784]
exp1_val   = [3.0364, 3.3325, 3.4906]

# EXPERIMENT 2: (LR=1e-5 Batch Size=8)
exp2_train = [3.4263, 2.7596, 2.5961]
exp2_val   = [3.1214, 2.9956, 2.9722]

# EXPERIMENT 3: (LR=1e-5 Batch Size=4)
exp3_train = [3.2313, 2.5955, 2.4221]
exp3_val   = [3.0453, 2.9462, 2.9395]

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Keep colors consistent across both panels
styles = {
    "exp1": dict(color="red", linestyle="--", marker="o"),
    "exp2": dict(color="blue", linestyle="--", marker="o"),
    "exp3": dict(color="green", linestyle="-", marker="o"),
}

# Training
ax1.plot(epochs, exp1_train, **styles["exp1"],
         label="Exp1: LR=1e-4, BS=8")

ax1.plot(epochs, exp2_train, **styles["exp2"],
         label="Exp2: LR=1e-5, BS=8")

ax1.plot(epochs, exp3_train, **styles["exp3"], linewidth=2,
         label="Exp3: LR=1e-5, BS=4")

ax1.set_title("Training Loss — LoRA runs", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Validation
ax2.plot(epochs, exp1_val, color="red", linestyle="--", marker="x",
         label="Exp1: LR=1e-4, BS=8 (overfit)")

ax2.plot(epochs, exp2_val, color="blue", linestyle="--", marker="x",
         label="Exp2: LR=1e-5, BS=8 (stable)")

ax2.plot(epochs, exp3_val, color="green", linestyle="-", marker="x", linewidth=2,
         label="Exp3: LR=1e-5, BS=4 (stable, best)")


ax2.set_title("Validation Loss — LoRA runs", fontsize=14)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()


plt.tight_layout()

# Save BEFORE show (safer)
fig.savefig("task4_lora_training_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved as task4_lora_training_comparison.png")
