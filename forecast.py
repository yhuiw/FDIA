import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

N_NODES = 187
DATA_DIRS = ['./v1', './v2', './v3', './v4', './v5']
TARGET_NODES = [86, 106, 111, 113, 114, 124, 126, 130]  # 0-indexed
SEQ_LEN = 24        # window: 6 hrs
PRED_LEN = 1
EP = 50
BATCH_SIZE = 128
LR = 5e-4           ##

def load_day_load(filepath, use_attack=False):
    """extract load time series from pkl"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    key = 'attack_pi' if use_attack else 'sch_pi'
    pi = data[key]
    steps = sorted(pi.keys(), key=lambda x: int(x[1:]))
    load = np.array([pi[k] for k in steps]).T
    return load.astype(np.float32)


def load_all_loads(dirs, use_attack=False):
    """load all days"""
    all_loads = []
    for d in dirs:
        if not os.path.exists(d):
            print(f"Warning: {d} not found, skipping")
            continue
        files = sorted([f for f in os.listdir(d) if f.endswith('.pkl')])
        for f in tqdm(files, desc=f'Loading {os.path.basename(d)}'):
            load = load_day_load(os.path.join(d, f), use_attack)
            all_loads.append(load)
    return all_loads


def create_sequences(loads, seq_len, pred_len, target_nodes=None):
    """sequences for LSTM"""
    X_list, y_list = [], []

    for load in loads:
        n_nodes, n_steps = load.shape
        if target_nodes is None:
            target_idx = list(range(n_nodes))
        else:
            target_idx = target_nodes

        for t in range(n_steps - seq_len - pred_len + 1):
            X_list.append(load[:, t:t+seq_len].T)
            y_list.append(load[target_idx, t+seq_len:t+seq_len+pred_len].T)

    return np.array(X_list), np.array(y_list)


def build_lstm(seq_len, n_feat, n_targets, pred_len):
    inp = layers.Input(shape=(seq_len, n_feat))

    x = layers.LSTM(256, return_sequences=True)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_targets * pred_len)(x)
    out = layers.Reshape((pred_len, n_targets))(out)

    return keras.Model(inputs=inp, outputs=out)


def predict_day(model, day_load, seq_len, target_nodes, mean, std, y_mean, y_std):
    """predict full day step-by-step"""
    n_steps = day_load.shape[1]
    day_norm = (day_load - mean) / std

    preds, trues = [], []
    for t in range(n_steps - seq_len):
        x = day_norm[:, t:t+seq_len].T[np.newaxis, ...]
        pred_norm = model.predict(x, verbose=0)
        pred = pred_norm * y_std + y_mean
        preds.append(pred[0, 0, :])
        trues.append(day_load[target_nodes, t+seq_len])

    return np.array(preds), np.array(trues)


if __name__ == '__main__':
    print("Loading scheduled load data...")
    all_loads = load_all_loads(DATA_DIRS, use_attack=False)
    print(f"Loaded {len(all_loads)} days, shape per day: {all_loads[0].shape}")

    # train/test split
    n_train = int(len(all_loads) * 0.75)
    train_loads = all_loads[:n_train]
    test_loads = all_loads[n_train:]
    print(f"Train days: {len(train_loads)}, Test days: {len(test_loads)}")

    print(f"\nCreating sequences (seq_len={SEQ_LEN})...")
    X_train, y_train = create_sequences(train_loads, SEQ_LEN, PRED_LEN, TARGET_NODES)
    X_test, y_test = create_sequences(test_loads, SEQ_LEN, PRED_LEN, TARGET_NODES)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # normalize
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    print("\nBuilding LSTM model...")
    model = build_lstm(SEQ_LEN, N_NODES, len(TARGET_NODES), PRED_LEN)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='mse',
        metrics=['mae']
    )
    model.summary()

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]

    print("\nTraining...")
    history = model.fit(X_train_norm, y_train_norm,
        validation_split=0.1, epochs=EP, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
    )

    # evaluate
    print("\nEvaluating on test set...")
    y_pred_norm = model.predict(X_test_norm, verbose=0)
    y_pred = y_pred_norm * y_std + y_mean

    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    fig1, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for day_idx, ax in enumerate(axes):
        if day_idx >= len(test_loads):
            break
        preds_day, trues_day = predict_day(
            model, test_loads[day_idx], SEQ_LEN, TARGET_NODES, mean, std, y_mean, y_std
        )
        # plot first target node
        node_i = 0
        time_axis = np.arange(len(trues_day)) * 0.25  # hours
        ax.plot(time_axis, trues_day[:, node_i], 'b-', label='True', linewidth=1.5)
        ax.plot(time_axis, preds_day[:, node_i], 'r--', label='Predicted', linewidth=1.5)
        ax.set_ylabel(f'Load (Node {TARGET_NODES[node_i]})')
        ax.set_title(f'Test Day {day_idx + 1}')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (hours)')
    fig1.suptitle('Load Forecast: True vs Predicted (Single Node)', fontsize=14)
    plt.tight_layout()
    plt.savefig('load_curves_days.png', dpi=150)
    print("Saved: load_curves_days.png")

    # ========== PLOT 2: One day, multiple nodes ==========
    fig2, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    test_day_idx = 5  # pick a day
    preds_day, trues_day = predict_day(
        model, test_loads[test_day_idx], SEQ_LEN, TARGET_NODES, mean, std, y_mean, y_std
    )
    time_axis = np.arange(len(trues_day)) * 0.25

    for i, ax in enumerate(axes):
        if i >= len(TARGET_NODES):
            ax.axis('off')
            continue
        ax.plot(time_axis, trues_day[:, i], 'b-', label='True', linewidth=1.2)
        ax.plot(time_axis, preds_day[:, i], 'r--', label='Predicted', linewidth=1.2)
        ax.set_title(f'Node {TARGET_NODES[i]}')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Load')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    fig2.suptitle(f'Load Forecast - Test Day {test_day_idx + 1} (All Target Nodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig('load_curves_nodes.png', dpi=150)
    print("Saved: load_curves_nodes.png")

    fig3, axes = plt.subplots(2, 2, figsize=(12, 8))

    # scatter: pred vs true
    ax = axes[0, 0]
    ax.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.05, s=1)
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel('True Load')
    ax.set_ylabel('Predicted Load')
    ax.set_title('Predicted vs True')
    ax.grid(alpha=0.3)

    # residual histogram
    ax = axes[0, 1]
    residuals = (y_pred - y_test).flatten()
    ax.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (std={residuals.std():.4f})')
    ax.grid(alpha=0.3)

    # per-node MAE
    ax = axes[1, 0]
    node_mae = np.mean(np.abs(y_pred - y_test), axis=(0, 1))
    ax.bar(range(len(TARGET_NODES)), node_mae)
    ax.set_xticks(range(len(TARGET_NODES)))
    ax.set_xticklabels(TARGET_NODES)
    ax.set_xlabel('Node')
    ax.set_ylabel('MAE')
    ax.set_title('MAE per Target Node')
    ax.grid(alpha=0.3, axis='y')

    # training history
    ax = axes[1, 1]
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('load_forecast_analysis.png', dpi=150)
    print("Saved: load_forecast_analysis.png")

    fig4, ax = plt.subplots(figsize=(12, 5))

    # take first 24 steps (6 hours) of a test day
    zoom_steps = 24
    node_i = 0
    ax.plot(time_axis[:zoom_steps], trues_day[:zoom_steps, node_i], 'b-o',
            label='True', linewidth=2, markersize=4)
    ax.plot(time_axis[:zoom_steps], preds_day[:zoom_steps, node_i], 'r--s',
            label='Predicted', linewidth=2, markersize=4)
    ax.fill_between(time_axis[:zoom_steps],
                    trues_day[:zoom_steps, node_i],
                    preds_day[:zoom_steps, node_i],
                    alpha=0.2, color='gray')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Load')
    ax.set_title(f'Load Forecast (Zoomed) - Node {TARGET_NODES[node_i]}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('load_curves_zoomed.png', dpi=150)
    print("Saved: load_curves_zoomed.png")

    plt.show()