import matplotlib.pyplot as plt

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Keep colors consistent across both panels
styles = {
    "exp1": dict(color="red", linestyle="--", marker="o"),
    "exp2": dict(color="blue", linestyle="--", marker="o"),
    "exp3": dict(color="green", linestyle="-", marker="o"),
}

# Training
ax1.plot(epochs, exp1_train, **styles["exp1"], label="Exp1: LR=1e-4, BS=8 (train loss ↓ fastest)")
ax1.plot(epochs, exp2_train, **styles["exp2"], label="Exp2: LR=1e-5, BS=8 (train loss ↓ slower)")
ax1.plot(epochs, exp3_train, **styles["exp3"], linewidth=2, label="Exp3: LR=1e-5, BS=4 (train loss ↓ slower)")

ax1.set_title("Training Loss — LoRA runs", fontsize=14)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Validation
ax2.plot(epochs, exp1_val, color="red", linestyle="--", marker="x",
         label="Exp1: LR=1e-4, BS=8 (early overfitting)")

ax2.plot(epochs, exp2_val, color="blue", linestyle="--", marker="x",
         label="Exp2: LR=1e-5, BS=8 (stable validation improvement)")

ax2.plot(epochs, exp3_val, color="green", linestyle="-", marker="x", linewidth=2,
         label="Exp3: LR=1e-5, BS=4 (most stable validation trend)")


ax2.set_title("Validation Loss — LoRA runs", fontsize=14)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

# Highlight GLOBAL best across all experiments
all_vals = [("Exp1", exp1_val), ("Exp2", exp2_val), ("Exp3", exp3_val)]
best_exp, best_epoch, best_loss = None, None, float("inf")
for name, vals in all_vals:
    m = min(vals)
    if m < best_loss:
        best_loss = m
        best_exp = name
        best_epoch = vals.index(m) + 1

ax2.scatter([best_epoch], [best_loss], s=140, marker="*", color="black", zorder=5)
ax2.annotate(f"Global best: {best_exp} Ep{best_epoch} ({best_loss:.4f})",
             xy=(best_epoch, best_loss),
             xytext=(best_epoch + 0.15, best_loss + 0.15),
             arrowprops=dict(arrowstyle="->", color="black"))

plt.tight_layout()

# Save BEFORE show (safer)
fig.savefig("task4_lora_training_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved as task4_lora_training_comparison.png")
