# Datasets

Raw dataset files are **not** redistributed via this repository. Each
dataset is governed by its original licensor; please obtain the data through
the official channels below and place the files at the expected paths.

## Unipen

- Source : https://www.unipen.org/
- Subsets used in this work: **1A** (digits, 10 classes), **1B** (uppercase
  letters, 26 classes), **1C** (lowercase letters, 26 classes).
- License: governed by the Unipen consortium's terms of use.

### Expected layout

```
data/Unipen/
├── raw-train-data-1a.txt
├── raw-train-data-1b.txt
├── raw-train-data-1c.txt
├── raw-test-data-1a.txt
├── raw-test-data-1b.txt
├── raw-test-data-1c.txt
├── train-label-1a.txt
├── train-label-1b.txt
├── train-label-1c.txt
├── test-label-1a.txt
├── test-label-1b.txt
└── test-label-1c.txt
```

Each ``raw-{split}-data-{subset}.txt`` is a plain-text file with one
trajectory per line: ``2*T`` whitespace-separated floats that the loader
reshapes to ``(T, 2)``. The accompanying ``{split}-label-{subset}.txt``
contains one integer class label per line.

After placement, the loader ``utils/unipen_data.py::load_data(subset)``
returns ``(X_train, y_train, X_test, y_test)`` with ``X`` normalized to
``[-1, 1]`` and reshaped to ``(N, T=50, 2)``.

## CASIA-OLHWDB

- Source : http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
- Subset used in this work: **CASIA-OLHWDB 1.1** (isolated online
  handwritten Chinese characters).
- License: governed by CASIA's research-use license; an academic request is
  required.

### Expected layout

Place the unpacked ``.pot`` files under:

```
CASIA/casia/raw/
├── train/  (1.1train-gb1/...)
└── test/   (1.1test-gb1/...)
```

The loader ``utils/casia_data.py`` (a re-export of
``CASIA/src/{potreader,casia_dataset,casia_collate}``) reads the raw
``.pot``, drops the pen-up sentinel coordinates ``(-1, 0)`` and the
character-end coordinate ``(-1, -1)``, concatenates all strokes into a
single connected ``(T, 2)`` trajectory, and normalizes per-sample to
``[-1, 1]``. **Pen-up information is not exposed to either the recognizer
or the attack** — see the threat model section of the top-level README.

## What is in this directory in Git

Only ``.gitkeep`` placeholders, so the empty directories survive cloning.
``.gitignore`` excludes the raw ``.txt`` and ``.pot`` files.
