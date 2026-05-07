import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
CSV_PATH = os.path.join(REPO_ROOT, 'output_feature', 'output_feature.csv')
FIGURE_DIR = os.path.join(REPO_ROOT, 'figure_for_paper')
OUT_PATH_PNG = os.path.join(FIGURE_DIR, 'fig_four_panel_improvement.png')
OUT_PATH_EPS = os.path.join(FIGURE_DIR, 'fig_four_panel_improvement.eps')
Y_COL = 'CompressionImprovementPct'
FEATURE_COLS_10 = ['MaxMinBitWidthDiff', 'BitWidthMean', 'BitWidthVar', 'BitWidthStd', 'BitWidthDiffAbsMean', 'BitWidthMedian', 'BitWidthP90', 'BitWidthChangeCount', 'BitWidthRunMean', 'ZeroFrac']
FEATURES_2_STD_RANGE = ['BitWidthStd', 'MaxMinBitWidthDiff']

def plot_scatter_on_ax(ax, df, col, xlabel, title_fs, label_fs, tick_fs, panel_title, annot_fs):
    raw = df[[col, Y_COL]].dropna().astype({col: float, Y_COL: float})
    ax.scatter(raw[col], raw[Y_COL], s=12, alpha=0.35, c='steelblue', edgecolors='none')
    if len(raw) >= 10:
        xr = raw[col].values
        yr = raw[Y_COL].values
        r = np.corrcoef(xr, yr)[0, 1] if np.std(xr) > 0 and np.std(yr) > 0 else 0
        r2 = r * r
        ax.text(0.02, 0.98, f'ρ = {r:.3f}  R² = {r2:.3f}', transform=ax.transAxes, fontsize=annot_fs, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel('Improvement (%)', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.set_title(panel_title, fontsize=title_fs, loc='center', x=0.4)
    ax.grid(False)

def plot_pred_vs_actual_on_ax(ax, y_test, y_pred, panel_title, label_fs, tick_fs, title_fs, annot_fs):
    ax.scatter(y_test, y_pred, alpha=0.35, s=12, c='steelblue', edgecolors='none')
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1.2)
    ax.set_xlabel('Actual improvement (%)', fontsize=label_fs)
    ax.set_ylabel('Predicted improvement (%)', fontsize=label_fs)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.set_title(panel_title, fontsize=title_fs, loc='center', x=0.4)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rho = np.corrcoef(y_test, y_pred)[0, 1]
    txt = f' ρ = {rho:.3f} R² = {r2:.3f}'
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=annot_fs, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def main():
    title_fs = label_fs = tick_fs = annot_fs = 22
    df = pd.read_csv(CSV_PATH)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.subplots_adjust(wspace=0.03, hspace=0.3)
    plt.rcParams.update({'font.size': title_fs})
    plot_scatter_on_ax(axes[0, 0], df, 'BitWidthDiffAbsMean', 'mean of bit width delta', title_fs=title_fs, label_fs=label_fs, tick_fs=tick_fs, panel_title='(a) The impact of smoothness', annot_fs=annot_fs)
    plot_scatter_on_ax(axes[0, 1], df, 'MaxMinBitWidthDiff', 'bit length range', title_fs=title_fs, label_fs=label_fs, tick_fs=tick_fs, panel_title='(b) The impact of bit length range', annot_fs=annot_fs)
    axes[0, 0].set_box_aspect(1)
    axes[0, 1].set_box_aspect(1)
    y = df[Y_COL].astype(float)
    X2 = df[FEATURES_2_STD_RANGE].astype(float).fillna(df[FEATURES_2_STD_RANGE].median())
    X10 = df[FEATURE_COLS_10].astype(float).fillna(df[FEATURE_COLS_10].median())
    idx_all = np.arange(len(df))
    idx_train, idx_rest = train_test_split(idx_all, test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=42)
    X_train2, X_test2 = (X2.iloc[idx_train], X2.iloc[idx_test])
    y_train2, y_test2 = (y.iloc[idx_train], y.iloc[idx_test])
    X_train10, X_test10 = (X10.iloc[idx_train], X10.iloc[idx_test])
    y_train10, y_test10 = (y.iloc[idx_train], y.iloc[idx_test])
    sc2 = StandardScaler()
    X_tr_s = sc2.fit_transform(X_train2)
    X_te_s = sc2.transform(X_test2)
    rf2 = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf2.fit(X_tr_s, y_train2)
    pred2 = rf2.predict(X_te_s)
    plot_pred_vs_actual_on_ax(axes[1, 0], y_test2, pred2, panel_title='(c) The prediction of 2 features', label_fs=label_fs, tick_fs=tick_fs, title_fs=title_fs, annot_fs=annot_fs)
    sc10 = StandardScaler()
    X_tr10 = sc10.fit_transform(X_train10)
    X_te10 = sc10.transform(X_test10)
    rf10 = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf10.fit(X_tr10, y_train10)
    pred10 = rf10.predict(X_te10)
    plot_pred_vs_actual_on_ax(axes[1, 1], y_test10, pred10, panel_title='(d) The prediction of 10 features', label_fs=label_fs, tick_fs=tick_fs, title_fs=title_fs, annot_fs=annot_fs)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(OUT_PATH_PNG, dpi=300, bbox_inches='tight')
    try:
        plt.savefig(OUT_PATH_EPS, format='eps', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f'EPS save failed: {e}')
    plt.close()
    print(f'Saved: {OUT_PATH_PNG}')
    print(f'Saved: {OUT_PATH_EPS}')
    print(f'(c) test n={len(y_test2)}, (d) test n={len(y_test10)}')
if __name__ == '__main__':
    main()
