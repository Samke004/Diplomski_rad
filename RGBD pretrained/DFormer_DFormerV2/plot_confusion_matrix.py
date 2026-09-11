"""
plot_confusion_matrix.py
────────────────────────────────────────────────────────────────────────────
Ucitava spremljenu matricu konfuzije (.npy) i generira cistu vizualizaciju
spremnu za znanstveni rad.

Ulaz: 4x4 matrica konfuzije (indeksi klasa: 0=pozadina, 1=deblo, 2=grane, 3=potpora)
Retci su STVARNE klase, stupci su PREDIKCIJE modela.

Izlazi tri varijante:
  confusion_matrix_fg_only.png     — 3x3, samo foreground klase (deblo/grane/potpora)
  confusion_matrix_fg_vs_all.png   — 3x4, foreground retci vs sve predikcije
  confusion_matrix_full.png        — 4x4, potpuna matrica

Usage:
  python plot_confusion_matrix.py \\
      --input confusion_matrix_DFormer-Base.npy \\
      --output_dir ./confusion_matrix_plots
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Nazivi klasa (redoslijed odgovara indeksima 0-3 u matrici)
CLASS_NAMES_FULL = ['Pozadina', 'Deblo', 'Grane', 'Potpora']
FOREGROUND_INDICES = [1, 2, 3]  # deblo, grane, potpora
FOREGROUND_NAMES = ['Deblo', 'Grane', 'Potpora']


def plot_matrix(cm, row_labels, col_labels, output_path,
                normalize=True, decimals=2):
    """
    Cista slika matrice konfuzije za tezni rad.
    Bez naslova - LaTeX opis figure to pokriva.

    normalize=True → normalizacija po recima (dijagonala = recall po klasi, u %)
    normalize=False → sirove brojke piksela
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1  # izbjegni dijeljenje s nulom
        cm_display = cm.astype(float) / row_sums * 100.0
        cbar_label = 'Postotak piksela u retku (%)'
        vmax = 100.0
    else:
        cm_display = cm.astype(float)
        cbar_label = 'Broj piksela'
        vmax = cm_display.max()

    # Dimenzije slike ovise o obliku matrice
    n_rows, n_cols = cm.shape
    fig_w = max(6.0, n_cols * 1.6)
    fig_h = max(5.0, n_rows * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm_display, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=11)

    # Ticks i labele
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)

    ax.set_xlabel('Predikcija modela', fontsize=12, labelpad=10)
    ax.set_ylabel('Referentna oznaka', fontsize=12, labelpad=10)

    # Vrijednosti u celijama
    thresh = cm_display.max() / 2.0
    for i in range(n_rows):
        for j in range(n_cols):
            value = cm_display[i, j]
            if normalize:
                # zamjena decimalne tocke za zarez radi hrvatskog stila
                text = f'{value:.{decimals}f}'.replace('.', ',') + ' %'
            else:
                # separator tisucica sa razmakom
                text = f'{int(value):,}'.replace(',', ' ')
            color = 'white' if cm_display[i, j] > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Sacuvano: {output_path}')


def compute_metrics(cm, class_names):
    """Izracunaj recall, precizaciju, F1 i IoU po klasi iz matrice konfuzije."""
    print(f'\n  {"Klasa":10s}  {"TP":>12s}  {"FP":>12s}  {"FN":>12s}  '
          f'{"Recall":>8s}  {"Precision":>10s}  {"IoU":>8s}')
    print('  ' + '-' * 84)
    ious = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ious.append(iou)

        print(f'  {name:10s}  {tp:>12,}  {fp:>12,}  {fn:>12,}  '
              f'{recall*100:>7.2f}%  {precision*100:>9.2f}%  {iou*100:>7.2f}%'
              .replace(',', ' '))
    mean_iou = np.mean(ious) * 100
    print(f'\n  mIoU po prikazanim klasama: {mean_iou:.2f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='Putanja do .npy matrice konfuzije (4x4)')
    parser.add_argument('--output_dir', default='./confusion_matrix_plots',
                        help='Direktorij u koji se spremaju izlazi')
    args = parser.parse_args()

    # Ucitaj punu 4x4 matricu
    cm_full = np.load(args.input)
    if cm_full.shape != (4, 4):
        raise ValueError(f'Ocekivano 4x4, dobiveno {cm_full.shape}')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Ucitana matrica konfuzije iz: {args.input}')
    print(f'Ukupno piksela: {cm_full.sum():,}'.replace(',', ' '))

    # ── Varijanta 1: 3x3 (samo foreground vs foreground) ────────────────────
    print('\n[1/3] Foreground vs foreground (3x3)')
    cm_fg = cm_full[np.ix_(FOREGROUND_INDICES, FOREGROUND_INDICES)]
    compute_metrics(cm_fg, FOREGROUND_NAMES)
    plot_matrix(
        cm_fg,
        row_labels=FOREGROUND_NAMES,
        col_labels=FOREGROUND_NAMES,
        output_path=os.path.join(args.output_dir, 'confusion_matrix_fg_only.png'),
        normalize=True,
    )

    # ── Varijanta 2: 3x4 (foreground retci vs sve predikcije) ───────────────
    print('\n[2/3] Foreground retci vs sve predikcije (3x4)')
    cm_fg_all = cm_full[FOREGROUND_INDICES, :]
    plot_matrix(
        cm_fg_all,
        row_labels=FOREGROUND_NAMES,
        col_labels=CLASS_NAMES_FULL,
        output_path=os.path.join(args.output_dir, 'confusion_matrix_fg_vs_all.png'),
        normalize=True,
    )

    # ── Varijanta 3: puna 4x4 matrica ───────────────────────────────────────
    print('\n[3/3] Puna 4x4 matrica')
    compute_metrics(cm_full, CLASS_NAMES_FULL)
    plot_matrix(
        cm_full,
        row_labels=CLASS_NAMES_FULL,
        col_labels=CLASS_NAMES_FULL,
        output_path=os.path.join(args.output_dir, 'confusion_matrix_full.png'),
        normalize=True,
    )

    # Spremi i CSV za mogucu naknadnu analizu
    csv_path = os.path.join(args.output_dir, 'confusion_matrix_full.csv')
    with open(csv_path, 'w') as f:
        f.write(',' + ','.join(CLASS_NAMES_FULL) + '\n')
        for i, name in enumerate(CLASS_NAMES_FULL):
            row = ','.join(str(int(x)) for x in cm_full[i])
            f.write(f'{name},{row}\n')
    print(f'\n  Sacuvano: {csv_path}')

    print(f'\nGotovo. Sve slike u: {args.output_dir}')


if __name__ == '__main__':
    main()
