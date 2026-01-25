import matplotlib.pyplot as plt
import numpy as np


# --- Plot function ---
def plot_comparison_report(all_histories):
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['r', 'b', 'g', 'c', 'm']

    for i, (name, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist['train_loss']) + 1)
        color = colors[i % len(colors)]

        # Grafico a sinistra: Training Loss
        ax[0].plot(epochs, hist['train_loss'], f'{color}o--', label=f'{name} (Train)')
        ax[0].set_title('Training Loss (Lower is better)')

        # Grafico a destra: Validation Loss
        ax[1].plot(epochs, hist['val_loss'], f'{color}x-', label=f'{name} (Val)')
        ax[1].set_title('Validation Loss (Overfitting detection)')

    for a in ax:
        a.set_xlabel('Epochs')
        a.set_ylabel('Loss')
        a.legend()
        a.grid(True, linestyle='--', alpha=0.6)

    plt.show()

def plot_single_experiment(history, title="Experiment Results"):
    """Visualizza Loss e PCK standard (0.1) per un singolo screening."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    ax[0].plot(epochs, history['train_loss'], 'go-', label='Train Loss')
    ax[0].plot(epochs, history['val_loss'], 'ro-', label='Val Loss')
    ax[0].set_title('Loss Progress')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    ax[0].grid(True, linestyle='--')

    # PCK (Nota: cambiato val_pck in pck_10)
    ax[1].plot(epochs, history['pck_10'], 'bo-', label='Val PCK@0.1')
    ax[1].set_title('Accuracy Progress (PCK @0.1)')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('%')
    ax[1].grid(True, linestyle='--')

    best_idx = np.argmax(history['pck_10'])
    best_pck = history['pck_10'][best_idx]
    ax[1].annotate(f'Best: {best_pck:.2f}%',
                   xy=(best_idx + 1, best_pck),
                   xytext=(best_idx + 1, best_pck - 5),
                   arrowprops=dict(facecolor='black', shrink=0.05))

    plt.suptitle(title)
    plt.show()

def plot_layer_comparison(all_results):
    """Confronta il PCK standard (0.1) di tutti i layer testati."""
    plt.figure(figsize=(10, 6))
    for layer_name, history in all_results.items():
        # Nota: cambiato val_pck in pck_10
        plt.plot(range(1, len(history['pck_10']) + 1),
                 history['pck_10'],
                 marker='o',
                 label=f'PCK {layer_name}')

    plt.title('Layer Screening: Validation PCK (@0.1) comparison')
    plt.xlabel('Epochs')
    plt.ylabel('PCK @ 0.1 (%)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

def plot_multi_threshold_pck(history, title="PCK Progress"):
    epochs = range(1, len(history['pck_10']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['pck_05'], 'r^--', label='PCK@0.05 (Fine)')
    plt.plot(epochs, history['pck_10'], 'bo-', label='PCK@0.10 (Standard)')
    plt.plot(epochs, history['pck_20'], 'gs-.', label='PCK@0.20 (Coarse)')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('PCK %')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Funzione per plottare il confronto tra più esperimenti
def plot_comparison(experiments):
    plt.figure(figsize=(10, 6))
    for name, hist in experiments.items():
        # Aggiornato da val_pck a pck_10
        plt.plot(range(1, len(hist['pck_10']) + 1), hist['pck_10'], label=name)

    plt.title('Comparison of Different Layers (Validation PCK)')
    plt.xlabel('Epochs')
    plt.ylabel('PCK @0.1 %')
    plt.legend()
    plt.grid(True)
    plt.show()