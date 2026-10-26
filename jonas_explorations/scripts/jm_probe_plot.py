"""#jm 2026-10-19: summarise jm_probe_results.csv (channel encoding x tokenizer) as a table and a figure
for section 13 of the DAC lab notebook.  Usage: python jm_probe_plot.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = Path("/data/groups/bci/jonas/workspace/bci_descript-audio-codec/jonas_explorations/notes_jm/figs/nanogpt_probe.png")
ENC = ["none", "learned", "rope4d"]
ENC_LABEL = {"none": "original (no channel info)", "learned": "channel dictionary (wce)", "rope4d": "4D RoPE (t, x, y, z)"}
DS_LABEL = {"eeg_dac_multichannel": "10k_1X1024\n(V=1024)", "eeg_dac_noPost8192": "baseline\n(V=8192)",
            "eeg_dac_corrupt4v3": "corrupt4_v3\n(V=8192)", "eeg_dac_chris512": "Chris 512\n(V=512)"}
CHANCE = {"fist_lr": 50.0, "ssvep": 100 / 12}


def main():
    df_all = pd.read_csv(HERE.parent / "results" / "jm_probe_results.csv").drop_duplicates(["out_dir", "task", "readout"], keep="last")
    main_dirs = {f"out-jm/{d[len('eeg_dac_'):]}-{e}" for d in DS_LABEL for e in ENC}   # the 3 x 4 grid only
    df = df_all[df_all.out_dir.isin(main_dirs)]
    extra = df_all[~df_all.out_dir.isin(main_dirs) & df_all.out_dir.str.startswith("out-jm/")]
    if len(extra):   # masking comparison, longer / bigger-data runs: table only
        ex = extra.pivot_table(index="out_dir", columns=["task", "readout"], values="acc")
        ex.insert(0, "bits", extra.groupby("out_dir").bits.first())
        print("extra runs:\n" + ex.round(2).to_string() + "\n")
        ex.to_csv(HERE.parent / "results" / "jm_probe_extra.csv")
    datasets = [d for d in DS_LABEL if (df.dataset == d).any()]
    # table
    rows = []
    for d in datasets:
        for e in ENC:
            g = df[(df.dataset == d) & (df.channel_encoding == e)]
            if g.empty:
                continue
            r = dict(tokenizer=DS_LABEL[d].replace("\n", " "), encoding=e, bits=g.bits.iloc[0])
            for task in ("fist_lr", "ssvep"):
                for ro in ("gpt_state", "embedding", "gpt_state_xsub", "embedding_xsub"):  # _xsub = subject-disjoint folds
                    gg = g[(g.task == task) & (g.readout == ro)]
                    r[f"{task}_{ro}"] = gg.acc.iloc[0] if len(gg) else np.nan
            rows.append(r)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    tab.to_csv(HERE.parent / "results" / "jm_probe_summary.csv", index=False)
    # figure
    sys.path.insert(0, "/data/groups/bci/jonas/workspace/style_guide")
    from zyphra_finalized_plot_helpers import bottom_legend, new_figure, save_figure, style_axis
    fig, theme = new_figure("nanoGPT on EEG tokens: channel encoding x tokenizer",
                            "left: held-out masked-prediction loss (dotted: log2 V) - middle/right: accuracy from the "
                            "transformer's states, hatched: token embeddings only (dotted: chance)",
                            mode="light", figsize=(17, 6))
    axes = fig.subplots(1, 3); fig.subplots_adjust(top=0.8, bottom=0.26, left=0.05, right=0.99, wspace=0.25)
    cols = {"none": theme.muted, "learned": theme.blue, "rope4d": theme.orange}
    x = np.arange(len(datasets)); w = 0.26
    ax = axes[0]; style_axis(ax, mode="light", title="bits per masked token (val)", grid="y", title_size=11)
    for i, e in enumerate(ENC):
        ys = [tab[(tab.tokenizer == DS_LABEL[d].replace("\n", " ")) & (tab.encoding == e)].bits.mean() for d in datasets]
        ax.bar(x + (i - 1) * w, ys, w, color=cols[e], label=ENC_LABEL[e])
    for j, d in enumerate(datasets):
        V = int(DS_LABEL[d].split("V=")[1].rstrip(")"))
        ax.hlines(np.log2(V), j - 0.45, j + 0.45, color=theme.muted, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 14); ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in datasets], fontsize=8)
    for ax, task in zip(axes[1:], ("fist_lr", "ssvep")):
        style_axis(ax, mode="light", title=f"{task}: accuracy from the states (%)", grid="y", title_size=11)
        for i, e in enumerate(ENC):
            g = tab[tab.encoding == e].set_index("tokenizer")
            ys = [g[f"{task}_gpt_state"].get(DS_LABEL[d].replace("\n", " "), np.nan) for d in datasets]
            ye = [g[f"{task}_embedding"].get(DS_LABEL[d].replace("\n", " "), np.nan) for d in datasets]
            ax.bar(x + (i - 1) * w, ys, w, color=cols[e])
            ax.bar(x + (i - 1) * w, ye, w, fill=False, edgecolor=cols[e], hatch="///", linewidth=0.8)
        ax.axhline(CHANCE[task], color=theme.muted, linestyle=":", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in datasets], fontsize=8)
        ax.set_ylim(0, max(80 if task == "fist_lr" else 50, np.nanmax(tab[[f"{task}_gpt_state", f"{task}_embedding"]].values) * 1.15))
    h, l = axes[0].get_legend_handles_labels(); bottom_legend(fig, h, l, mode="light", y=0.02)
    save_figure(fig, FIG, mode="light", also_svg=False)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
