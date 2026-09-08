"""
Feature calculation and final-set picking (RPE features).

Port of the ML notebooks 1 / 1b / 6 / 5 ("FEATURE CALCULATION & EVALUATION")
into the pipeline, run as the last batch step after collect_data.

Reads:
    <output_dir>/collected_data/<stem>/L2_ratio50.csv
    <output_dir>/collected_data/spotify_all.csv          (optional)

Writes:
    <output_dir>/feature_sets/calculated_features/<set>_features.csv
        7 candidate sets (52 features total):
        density (12), syncopation_like (3), swing_8th (4), swing_16th (4),
        metrical_balance (3), consistency (1), statistical (19), syncopation (6)
    <output_dir>/feature_sets/final_features/our_features.csv
        the 25 features used for modelling (fixed theory-based drop list)
    <output_dir>/feature_sets/final_features/spotify_features.csv
        Spotify baseline features (if spotify_all.csv exists)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# IOI class labels as written by the pipeline (label-fix of 2026-07-31)
CATS = ['1/16', '1/8', '3/16', '1/4', '6/16', '2/4', '4/4']
SHARE_CATS = ['1/16', '1/8', '3/16', '1/4']  # 6/16, 2/4, 4/4 are quasi-constant

# Theory-based drop list (see thesis Section "Feature engineering"): applied to
# the 52 candidates, leaves the 25 modelled features of our_features.csv.
THEORY_DROP = [
    'syncopation_a_RP', 'syncopation_e_RP', 'note2_8th', 'note3_8th',
    'ext3_8th', 'density_RP_any', 'n_pairs_16th', 'note2_16th',
    'longest_run_16th', 'note3_16th', 'ext3_16th', 'swing_16th_ratio_min',
    'swing_8th_ratio_max', 'swing_8th_ratio_min', 'class_share_1_8',
    'share_16th', 'class_share_1_16', 'density_RP_strong',
    'total_grid_positions_str', 'strength_evenness', 'syncopation_and_GP',
    'swing_16th_ratio_max', 'swing_8th_ratio_std', 'class_share_1_4',
    'groove_pulse_strength', 'ioi_microtiming_complexity',
    'rp_centroid_strong',
]

SPOTIFY_FEATURES = [
    'SP_danceability', 'SP_energy', 'SP_loudness', 'SP_speechiness',
    'SP_acousticness', 'SP_instrumentalness', 'SP_liveness', 'SP_valence',
    'SP_tempo', 'SP_key', 'SP_mode', 'SP_time_signature',
]

# The 4 feature families of the final set (table order / label colours), used
# to order and colour the reduced correlation matrix.
FAMILIES = [
    ('Aggregated section-level statistics', '#2E6DB4', [
        'mean_section_tempo', 'microtiming_degree', 'microtiming_complexity',
        'ioi_microtiming_degree', 'pulse_strength', 'groove_ioi_pulse_strength',
        'class_share_3_16', 'ioi_entropy', 'rp_centroid_all', 'rp_centroid_weak',
        'RP_microtiming_mean', 'BP_microtiming_mean']),
    ('Density', '#2E8B57', ['density_GP', 'n_pairs_8th', 'share_8th', 'share_quarter']),
    ('Swing', '#B7950B', ['swing_8th_ratio_avg', 'swing_16th_ratio_avg',
                          'swing_16th_ratio_std']),
    ('Syncopation', '#7D3C98', ['syncopation_e_GP', 'syncopation_and_RP',
                                'syncopation_a_GP', 'ea_contrast',
                                'has_16th_offbeats', 'ea_saturated']),
]


def _correlation_matrices(allfeat: pd.DataFrame, final: list, out_dir: Path):
    """Spearman matrices: full candidate pool (clustered) + final set (by family)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    candidates = [c for c in allfeat.columns if c != 'song_id']

    # ---- full candidate matrix, ordered by hierarchical clustering ----------
    sp = allfeat[candidates].corr(method='spearman')
    dist = (1 - sp.abs()).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    Z = linkage(squareform(dist.values, checks=False), method='average')
    spo = sp.iloc[leaves_list(Z), leaves_list(Z)]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(spo, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Spearman rho', 'shrink': 0.6}, ax=ax)
    ax.tick_params(labelsize=9)
    ax.set_title(f'RPE feature set — Spearman correlation ({len(candidates)} features)',
                 fontsize=16, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'spearman_matrix_full.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- reduced matrix, grouped by the 4 feature families ------------------
    feat2grp = {f: g for g, _, feats in FAMILIES for f in feats}
    grpcolor = {g: c for g, c, _ in FAMILIES}
    order = [f for _, _, feats in FAMILIES for f in feats if f in final]
    order += [f for f in final if f not in order]        # safety net
    spf = allfeat[final].corr(method='spearman').loc[order, order]
    fig, ax = plt.subplots(figsize=(13.5, 10))
    sns.heatmap(spf, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Spearman rho', 'shrink': 0.55}, ax=ax)
    ax.tick_params(labelsize=11)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        grp = feat2grp.get(lab.get_text())
        if grp:
            lab.set_color(grpcolor[grp])
            lab.set_fontweight('bold')
    sizes = [sum(1 for f in feats if f in final) for _, _, feats in FAMILIES]
    for x in np.cumsum([sz for sz in sizes if sz > 0])[:-1]:
        ax.axhline(x, color='#2C3E50', lw=1.4)
        ax.axvline(x, color='#2C3E50', lw=1.4)
    handles = [Patch(facecolor=c, edgecolor='none', label=g) for g, c, _ in FAMILIES]
    ax.legend(handles=handles, title='Feature group', loc='upper right',
              bbox_to_anchor=(-0.01, -0.02), fontsize=10.5, title_fontsize=11.5,
              frameon=False, handlelength=1.2, handleheight=1.2)
    ax.set_title(f'RPE feature set — Spearman correlation (reduced, {len(final)} features)',
                 fontsize=15.5, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'spearman_matrix_reduced.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  spearman_matrix_full.png / spearman_matrix_reduced.png written')


def _load_source(collected_csv: Path) -> pd.DataFrame:
    """Load the collected L2_ratio50 grid, 4/4 only, one row per song."""
    rh = pd.read_csv(collected_csv)
    rh = rh[rh['time_signature'] == 4].copy()
    ndup = rh['song_id'].duplicated().sum()
    if ndup:
        ids = sorted(rh.loc[rh['song_id'].duplicated(keep=False), 'song_id'].unique())
        print(f'  ! dropping {ndup} duplicate song_id row(s) {ids} (keeping first)')
        rh = rh.drop_duplicates('song_id', keep='first')
    return rh.reset_index(drop=True)


def _row_mean(vals, mask):
    """Per-row mean of vals over mask; NaN where the mask never holds."""
    m = mask & ~np.isnan(vals)
    cnt = m.sum(1)
    s = np.where(m, vals, 0.0).sum(1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def calculate_features(output_dir: Path, stem: str = 'drums'):
    """Compute all candidate feature sets and write the final feature CSVs."""
    collected = Path(output_dir) / 'collected_data'
    src = collected / stem / 'L2_ratio50.csv'
    if not src.exists():
        print(f'  ! {src} not found - run collect_data first')
        return None

    feature_dir = Path(output_dir) / 'feature_sets'
    out_calc = feature_dir / 'calculated_features'
    out_final = feature_dir / 'final_features'
    out_calc.mkdir(parents=True, exist_ok=True)
    out_final.mkdir(parents=True, exist_ok=True)

    rh = _load_source(src)
    n = len(rh)
    pos = np.arange(32)
    eps = 1e-9
    sid = rh['song_id'].values
    print(f'  songs: {n} (4/4, deduplicated)')

    # ---- position matrices --------------------------------------------------
    GPs = rh[[f'GP_str_{i}' for i in range(32)]].to_numpy(float)   # graded strength
    GPm = rh[[f'GP_med_{i}' for i in range(32)]].to_numpy(float)
    GPi = rh[[f'GP_iqr_{i}' for i in range(32)]].to_numpy(float)
    RPs = rh[[f'RP_str_{i}' for i in range(32)]].to_numpy(float)   # ternary 0/.5/1
    RPm = rh[[f'RP_med_{i}' for i in range(32)]].to_numpy(float)
    occ = RPs > 0

    assert f'GPB_str_{CATS[-1]}' in rh.columns, (
        'GPB_str_4/4 missing -- this CSV predates the IOI label-fix (2026-07-31)')
    GPBs = rh[[f'GPB_str_{c}' for c in CATS]].to_numpy(float)
    GPBm = rh[[f'GPB_med_{c}' for c in CATS]].to_numpy(float)
    GPBi = rh[[f'GPB_iqr_{c}' for c in CATS]].to_numpy(float)
    BPs = rh[[f'BP_str_{c}' for c in CATS]].to_numpy(float)
    BPm = rh[[f'BP_med_{c}' for c in CATS]].to_numpy(float)

    # ---- SET: density (+ pair / run features) -------------------------------
    Q, E8, E16 = pos % 4 == 0, pos % 4 == 2, pos % 2 == 1

    def n_pairs(step):
        c = np.zeros(n, int)
        for m in range(32 // (2 * step)):
            p0 = (2 * step * m) % 32
            c += (occ[:, p0] & occ[:, (p0 + step) % 32]
                  & occ[:, (p0 + 2 * step) % 32]).astype(int)
        return c

    def runs(step, k):
        c = np.zeros(n, int)
        for p0 in range(0, 32, step):
            m = np.ones(n, bool)
            for j in range(k + 1):
                m &= occ[:, (p0 + j * step) % 32]
            c += m.astype(int)
        return c

    def longest_run(o):
        d = np.concatenate([o, o])
        best = cur = 0
        for x in d:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return min(best, 32)

    density = pd.DataFrame({'song_id': sid})
    density['density_GP'] = (GPs > 0).sum(1) / 32
    density['density_RP_any'] = (RPs > 0).sum(1) / 32
    density['density_RP_strong'] = (RPs == 1).sum(1) / 32
    density['n_pairs_8th'] = n_pairs(2)
    density['n_pairs_16th'] = n_pairs(1)
    density['note2_8th'] = runs(2, 2)
    density['note3_8th'] = runs(2, 3)
    density['note2_16th'] = runs(1, 2)
    density['note3_16th'] = runs(1, 3)
    density['ext3_8th'] = runs(2, 3) / (runs(2, 2) + eps)
    density['ext3_16th'] = runs(1, 3) / (runs(1, 2) + eps)
    density['longest_run_16th'] = [longest_run(o) for o in occ]

    # ---- SET: syncopation_like (ea contrast + flags) ------------------------
    E, A = pos % 4 == 1, pos % 4 == 3
    e = RPs[:, E].mean(1)
    a = RPs[:, A].mean(1)
    s = e + a
    defined = s > 0
    ea = np.where(defined, (e - a) / np.where(defined, s, 1), np.nan)

    syncopation_like = pd.DataFrame({'song_id': sid})
    syncopation_like['ea_contrast'] = ea
    syncopation_like['has_16th_offbeats'] = defined.astype(int)
    syncopation_like['ea_saturated'] = np.where(
        defined, (np.abs(ea) == 1).astype(float), np.nan)

    # ---- SET: swing_8th / swing_16th ----------------------------------------
    def swing_ratios(step):
        npairs_ = 32 // (2 * step)
        R = np.full((n, npairs_), np.nan)
        for m in range(npairs_):
            p0 = (2 * step * m) % 32
            p1 = (p0 + step) % 32
            p2 = (p0 + 2 * step) % 32
            d1 = step + (RPm[:, p1] - RPm[:, p0])
            d2 = step + (RPm[:, p2] - RPm[:, p1])
            v = (occ[:, p0] & occ[:, p1] & occ[:, p2]
                 & (d1 > 0.1 * step) & (d2 > 0.1 * step))
            R[v, m] = d1[v] / d2[v]
        return R

    def swing_set(step, tag):
        R = swing_ratios(step)
        nv = np.isfinite(R).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      # all-NaN rows -> NaN
            df = pd.DataFrame({'song_id': sid})
            df[f'swing_{tag}_ratio_avg'] = np.nanmean(R, 1)
            df[f'swing_{tag}_ratio_std'] = np.where(
                nv >= 2, np.nanstd(R, 1, ddof=1), np.nan)
            df[f'swing_{tag}_ratio_max'] = np.where(
                nv >= 1, np.nanmax(np.where(np.isfinite(R), R, -np.inf), 1), np.nan)
            df[f'swing_{tag}_ratio_min'] = np.where(
                nv >= 1, np.nanmin(np.where(np.isfinite(R), R, np.inf), 1), np.nan)
        return df

    swing_8th = swing_set(2, '8th')
    swing_16th = swing_set(1, '16th')

    # ---- SET: metrical_balance / consistency --------------------------------
    q = GPs[:, Q].mean(1)
    e8 = GPs[:, E8].mean(1)
    e16 = GPs[:, E16].mean(1)
    tot = q + e8 + e16 + eps
    metrical_balance = pd.DataFrame({'song_id': sid})
    metrical_balance['share_8th'] = e8 / tot
    metrical_balance['share_16th'] = e16 / tot
    metrical_balance['share_quarter'] = q / tot

    def evenness(row):
        v = row[row > 0]
        if len(v) < 2:
            return np.nan
        p = v / v.sum()
        return float(-(p * np.log(p)).sum() / np.log(len(v)))

    consistency = pd.DataFrame({'song_id': sid})
    consistency['strength_evenness'] = [evenness(r) for r in GPs]

    # ---- SET: statistical ---------------------------------------------------
    gp_on = GPs > 0
    gpb_on = GPBs > 0
    BEATS = np.arange(0, 32, 4)
    POS = np.tile(pos, (n, 1)).astype(float)

    statistical = pd.DataFrame({'song_id': sid})
    statistical['microtiming_degree'] = _row_mean(np.abs(GPm), gp_on)
    statistical['microtiming_complexity'] = _row_mean(GPi, gp_on)
    statistical['pulse_strength'] = GPs[:, BEATS].mean(1)
    statistical['ioi_microtiming_degree'] = _row_mean(np.abs(GPBm), gpb_on)
    statistical['ioi_microtiming_complexity'] = _row_mean(GPBi, gpb_on)
    statistical['groove_pulse_strength'] = _row_mean(GPs, gp_on)
    statistical['groove_ioi_pulse_strength'] = _row_mean(GPBs, gpb_on)
    statistical['mean_section_tempo'] = rh['mean_section_tempo'].values
    statistical['total_grid_positions_str'] = GPs.sum(1)
    tot_gpb = GPBs.sum(1)
    P = np.where(tot_gpb[:, None] > 0,
                 GPBs / np.where(tot_gpb[:, None] > 0, tot_gpb[:, None], 1.0),
                 np.nan)
    for c in SHARE_CATS:
        statistical[f'class_share_{c.replace("/", "_")}'] = P[:, CATS.index(c)]
    plogp = np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0)
    statistical['ioi_entropy'] = np.where(tot_gpb > 0, -plogp.sum(1), np.nan)
    w = RPs.sum(1)
    statistical['rp_centroid_all'] = np.where(
        w > 0, (RPs * pos).sum(1) / np.maximum(w, eps), np.nan) / 32
    statistical['rp_centroid_strong'] = _row_mean(POS, RPs == 1.0) / 32
    statistical['rp_centroid_weak'] = _row_mean(POS, RPs == 0.5) / 32
    statistical['RP_microtiming_mean'] = _row_mean(RPm, RPs > 0)
    statistical['BP_microtiming_mean'] = _row_mean(BPm, BPs > 0)

    # ---- SET: syncopation (offbeat strength sums) ---------------------------
    e_pos = [1, 5, 9, 13, 17, 21, 25, 29]
    and_pos = [2, 6, 10, 14, 18, 22, 26, 30]
    a_pos = [3, 7, 11, 15, 19, 23, 27, 31]
    specs = [
        ('RP_str', e_pos, 'syncopation_e_RP'),
        ('GP_str', e_pos, 'syncopation_e_GP'),
        ('RP_str', and_pos, 'syncopation_and_RP'),
        ('GP_str', and_pos, 'syncopation_and_GP'),
        ('RP_str', a_pos, 'syncopation_a_RP'),
        ('GP_str', a_pos, 'syncopation_a_GP'),
    ]
    syncopation = pd.DataFrame({'song_id': sid})
    for prefix, positions, name in specs:
        cols = [f'{prefix}_{p}' for p in positions if f'{prefix}_{p}' in rh.columns]
        syncopation[name] = rh[cols].sum(axis=1).values

    # ---- write one CSV per set ----------------------------------------------
    sets = {
        'density': density,
        'syncopation_like': syncopation_like,
        'swing_8th': swing_8th,
        'swing_16th': swing_16th,
        'metrical_balance': metrical_balance,
        'consistency': consistency,
        'statistical': statistical,
        'syncopation': syncopation,
    }
    total = 0
    for name, dfset in sets.items():
        feats = [c for c in dfset.columns if c != 'song_id']
        dfset[['song_id'] + feats].to_csv(out_calc / f'{name}_features.csv', index=False)
        total += len(feats)
        print(f'  {name:<20} {len(feats):>3} features -> {name}_features.csv')
    print(f'  candidate pool: {total} features across {len(sets)} sets')

    # ---- pick the final feature set -----------------------------------------
    allfeat = None
    for dfset in sets.values():
        allfeat = dfset if allfeat is None else allfeat.merge(
            dfset, on='song_id', how='outer')
    candidates = [c for c in allfeat.columns if c != 'song_id']
    final = [f for f in candidates if f not in THEORY_DROP]
    missing_drop = [f for f in THEORY_DROP if f not in candidates]
    if missing_drop:
        print(f'  ! THEORY_DROP names not in candidates: {missing_drop}')
    allfeat[['song_id'] + final].to_csv(out_final / 'our_features.csv', index=False)
    print(f'  our_features.csv: {len(final)} features x {len(allfeat)} songs')

    # ---- correlation matrices (full candidate pool + final set) -------------
    try:
        _correlation_matrices(allfeat, final, feature_dir)
    except Exception as e:
        print(f'  ! correlation matrices skipped: {e}')

    # ---- Spotify baseline features ------------------------------------------
    spotify_all = collected / 'spotify_all.csv'
    if spotify_all.exists():
        sp = pd.read_csv(spotify_all)
        present = [f for f in SPOTIFY_FEATURES if f in sp.columns]
        sp[['song_id'] + present].to_csv(out_final / 'spotify_features.csv', index=False)
        print(f'  spotify_features.csv: {len(present)} features x {len(sp)} songs')

    return out_final / 'our_features.csv'


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python feature_calculation.py /path/to/batch/output [stem]')
        sys.exit(1)
    calculate_features(Path(sys.argv[1]),
                       stem=sys.argv[2] if len(sys.argv) > 2 else 'drums')
