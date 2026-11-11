# Dataset — ESC-50 animal recordings

`audio/` · **360 WAV files** · 151.4 MiB · nine animal categories, 40 clips each

| | |
|---|---|
| Source | [ESC-50](https://github.com/karolpiczak/ESC-50), K. J. Piczak (2015) |
| Full dataset | 2,000 five-second environmental recordings in 50 classes |
| Used here | the 9 animal classes only |
| Format | 5 s · 44.1 kHz · mono · 16-bit PCM (441,044 bytes per file) |
| Licence | **CC BY-NC 3.0** — non-commercial use only |

`esc50.csv` is the complete ESC-50 metadata index (2,000 rows, unmodified):

```
sha256  ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246
```

## Columns

| Column | Meaning |
|---|---|
| `filename` | Clip file name, matching `audio/<filename>` |
| `fold` | ESC-50's official 5-fold cross-validation assignment |
| `target` | Class index 0–49 |
| `category` | Class name (`dog`, `cat`, `rooster`, …) |
| `esc10` | Whether the clip belongs to the easier ESC-10 subset |
| `src_file` | Freesound source recording id |
| `take` | Which take of that source recording |

## Which clips are here

The nine categories are:

```
cat, cow, crow, dog, frog, hen, pig, rooster, sheep
```

```python
metadata = pd.read_csv("esc50.csv")
subset = metadata[metadata["category"].isin(TARGET_ANIMALS)]   # 360 rows
```

All 360 rows of that subset are present — no clip referenced by the notebook is missing.

## Licence and attribution

ESC-50 is released under the **Creative Commons Attribution-NonCommercial 3.0 Unported**
licence; the full text is in [`LICENSE-CC-BY-NC-3.0.md`](LICENSE-CC-BY-NC-3.0.md).

Two consequences worth being explicit about:

- **Attribution is required.** The recordings come from
  [Freesound](https://freesound.org/); ESC-50 keeps a per-clip attribution list in its own
  `LICENSE` file, and the source recording id for every clip here is in `esc50.csv`
  (`src_file`).
- **Non-commercial only.** This audio may not be used in a commercial product or service.

If you use this dataset in academic work, cite:

> K. J. Piczak. *ESC: Dataset for Environmental Sound Classification*. Proceedings of the
> 23rd ACM Conference on Multimedia, Brisbane, Australia, 2015.
> [doi:10.1145/2733373.2806390](http://dx.doi.org/10.1145/2733373.2806390)

## Reproducing the subset

Download ESC-50 and copy the matching files:

```bash
git clone https://github.com/karolpiczak/ESC-50.git
cp ESC-50/audio/<filename> audio/
```

The notebook only ever reads `audio/<filename>` for the rows in the nine categories, so a
full 2,000-clip download is not required — but the class list must match, or the label
encoder will produce a different label order.
