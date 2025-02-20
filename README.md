<p align="center">
<img src="./docs/icon.svg" height="240px">
</p>

<p align="center">
A soft and fast pattern matcher for billion-scale corpora.
</p>

<p align="center">
<a href="https://pypi.org/project/softmatcha"><img alt="PyPi" src="https://img.shields.io/pypi/v/softmatcha"></a>
<a href="https://github.com/softmatcha/softmatcha/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/softmathca/sotmatcha.svg"></a>
<a href=""><img src="https://github.com/softmatcha/softmatcha/actions/workflows/ci.yaml/badge.svg"></a>
</p>
<p align="center">
<b>
      <a href="https://openreview.net/forum?id=Q6PAnqYVpo">Paper</a> |
      <a href="https://softmatcha.github.io">Website</a> |
      <a href="https://huggingface.co/softmatcha">Demo</a> |
      <!-- <a href="https://softmatcha.readthedocs.io">Reference docs</a> | -->
      <a href="https://github.com/softmatcha/softmatcha#citation">Citation</a>
</b>
</p>

## Installation

You can install via PyPi:

``` bash
pip install softmatcha
```

For the development purposes, you can install from the source via uv:

``` bash
git clone https://github.com/softmatcha/softmatcha.git
cd softmatcha/
uv sync
```

or pip:

``` bash
git clone https://github.com/softmatcha/softmatcha.git
cd softmatcha/
pip install -e ./
```

### MacOS
Before running `pip install`, you need to setup libraries and environment variables:
``` bash
brew install pkg-config icu4c
export CFLAGS="-std=c++11"
export PATH="$(brew --prefix)/opt/icu4c/bin:$(brew --prefix)/opt/icu4c/sbin:$PATH"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:$(brew --prefix)/opt/icu4c/lib/pkgconfig"
pip install softmatcha

```
## Quick start

SoftMatcha implements two search types: scan and index.

- Scan: search texts without indexing and any preprocessing like `grep`, which is useful for small corpora.
- Index: search texts with an index, effectively works on billion-scale corpora.

### Scan: `softmatcha-grep`

`softmatcha-grep` searches corpora without indexing:

``` shell-session
$ softmatcha-grep "the jazz musician" corpus.txt
```

The first arugment is the pattern string and the second one is a file or files to be searched.
The other arguments can be seen by `softmatcha-grep -h`.

### Index: `softmatcha-index` and `softmatcha-search`

`softmatcha-index` builds a search index from corpora:

``` shell-session
$ softmatcha-index --index corpus.idx corpus.txt
```

`softmatcha-search` quickly searches patterns with a search index:

``` shell-session
$ softmatcha-search --index corpus.idx "the jazz musician"
```

## Options

For development purposes,
- `--profile=true` measures the execution time.
- `--log` outputs the verbose information.

For searchers,
- `--backend {gensim,fasttext,transformers}`: Backend framework for embeddings.
- `--model <NAME>`: Name of word embeddings.
- `--threshold` specifies the threshold for soft matching.

For controlling outputs,
- `-n`, `--line_number` prints line number with output lines.
- `-o`, `--only_matching` outputs only matched patterns.

## List of implementations
### Embeddings
- [gensim](https://github.com/piskvorky/gensim)
- [fastText](https://github.com/facebookresearch/fastText)
- [transformers](https://github.com/huggingface/transformers) (embedding layers)

### Searchers
#### Scan: `softmatcha-grep`
- Naive search: `--search naive`
- Quick search (default): `--search quick`

#### Index: `softmatcha-index` and `softmatcha-search`
- Inverted index search

## Citation
If you use this software, please cite:

``` bibtex
@inproceedings{
  deguchi-iclr-2025-softmatcha,
  title={SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches},
  author={Deguchi, Hiroyuki and Kamoda, Go and Matsushita, Yusuke and Taguchi, Chihiro and Waga, Masaki and Suenaga, Kohei and Yokoi, Sho},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR 2025)},
  year={2025},
  url={https://openreview.net/forum?id=Q6PAnqYVpo}
}
```

## License

This software is mainly developed by [Hiroyuki
Deguchi](https://sites.google.com/view/hdeguchi) and published under the
MIT-license.
