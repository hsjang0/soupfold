# SoupFold🍲

<p align="center"><img src="assets/overview.png" width="100%"></p>

[![arXiv](https://img.shields.io/badge/arXiv-2609.15552-b31b1b.svg)](https://arxiv.org/abs/2609.15552)

Official code for **Co-folding with a Soup of Representations**
(Hyosoon Jang, Taewon Kim, Sungsoo Ahn), [arXiv:2609.15552](https://arxiv.org/abs/2609.15552).

## Abstract

Co-folding models such as AlphaFold3, Protenix, ESMFold2, and OpenDDE have advanced rapidly, yet no single model consistently performs best across all biomolecular complexes. In this paper, we show that their pair representations encode complementary information that can be transferred across models to improve structure prediction. We introduce SoupFold, which combines pair representations from multiple co-folding models in a common representation space and generates structures from the combined representation. Importantly, SoupFold does not retrain the co-folding models and learns only simple mappings to transfer representations across models. We evaluate SoupFold on antibody-antigen, protein-protein, protein-ligand, molecular glue, GPCR, and oligomeric complex prediction using AlphaFold3, Protenix, ESMFold2, and OpenDDE. By combining representations across models, SoupFold improves over individual co-folding models across the considered benchmarks.

## How it works

SoupFold improves a co-folding model by letting it borrow what other co-folding models see.
One model is the **anchor**. It folds the complex as usual up to its trunk, and its trunk pair
representation is then averaged with those of the other models, the **peers**. Each peer
representation is first translated into the anchor's space by a small learned map. The anchor
then runs its own diffusion and confidence heads on the averaged representation.

```math
H = \frac{w_{m^\star} H^{(m^\star)} + \sum_{m \in \mathcal{M} \setminus \{m^\star\}} w_m \, f_{m \to m^\star}\big(H^{(m)}\big)}{w_{m^\star} + \sum_{m \in \mathcal{M} \setminus \{m^\star\}} w_m}
```

$m^\star$ is the anchor and the other models in $\mathcal{M}$ are the peers. $H^{(m)}$ is the pair
representation of model $m$, normalised per channel, and $f_{m \to m^\star}$ is the learned map from
peer $m$ to the anchor. This code uses equal weights ($w_m = 1$). $H$ is scaled back to the anchor's
channel statistics before diffusion.
The maps and channel statistics are in `weights/`.

## Models

| model | version | role |
|---|---|---|
| AlphaFold3 | v3.0.1 | peer |
| Protenix | 2.0.0 | anchor, peer |
| OpenDDE | 1.1.0 | anchor, peer |
| ESMFold2 | esm 3.4.0 | anchor, peer |

Each model runs in its own Python environment. SoupFold does not ship or modify their code.
Representations are captured and injected at runtime by `soupfold/hooks/`.

> **IMPORTANT NOTE: apply the patches in `patches/` before running.**
> SoupFold averages representations token by token, so every model must split a complex into the
> same tokens. Out of the box they disagree in two places, and the patches fix that.
>
> | patch | package | change |
> |---|---|---|
> | `protenix_parser.diff` | Protenix | Amino acid ligands (e.g. free ILE) get per-atom tokens, as in AF3. |
> | `opendde_parser.diff` | OpenDDE | Same as Protenix. |
> | `af3_atom_layout.diff` | AlphaFold3 | Keeps the glycan O1 atom, as Protenix and OpenDDE do. |
>
> ```bash
> patch -p1 -d <protenix site-packages> < patches/protenix_parser.diff
> patch -p1 -d <opendde site-packages>  < patches/opendde_parser.diff
> patch -p1 -d <alphafold3 repo>        < patches/af3_atom_layout.diff
> ```
>
> Proteins without ligands or glycans tokenise identically either way. ESMFold2 needs no patch.

## Input

One Protenix-format JSON per system, shared by all four models. MSA and template paths may be
relative to the JSON file. `examples/9mnb/` is a ready-to-run antibody example from ARK-AB
(two Fabs bound to one antigen) with its MSAs, templates and template structures.

```json
[{"name": "9mnb", "sequences": [
  {"proteinChain": {"sequence": "EVQL...", "count": 1, "unpairedMsaPath": "msa/chain1_unpaired.a3m",
                    "pairedMsaPath": "msa/chain1_paired.a3m", "templatesPath": "templates/chain1_hmmsearch.a3m"}},
  ...]}]
```

## Quick start

Copy `config.example.json` to `config.json` and fill in each model's Python environment,
weights and data files. Then run:

```bash
python scripts/soupfold.py --config config.json --input examples/9mnb/9mnb.json
```

The pipeline has three steps:

1. **Native runs.** All four models run on the input and store their trunk representations.
2. **Token layouts.** Each model's token order is recorded so the representations can be aligned.
3. **SoupFold.** Protenix, OpenDDE and ESMFold2 each act as anchor with the other three as peers.

You choose what the native runs produce:

- **Representations only** (default). Native runs draw one sample with 2 diffusion steps, only to
  get the representations. Those structures are not meaningful and go to `logs/`.
- **Representations and native structures** (`--sample`). Native runs also sample 5 structures
  with each model's default steps into `samples/standalone/`, for comparison with SoupFold.

SoupFold itself always draws 5 samples with the anchor's default steps. Use `--seeds 1-5` for
several seeds and `--anchors` to choose anchors. We rank samples by the anchor's own confidence.

## Outputs

Everything goes under `--workdir` (default `soupfold_out/`):

| directory | content |
|---|---|
| `samples/soupfold/<anchor>/seed_<k>/` | SoupFold structures (`<system>__s<i>.cif`) and confidences (`<system>__conf.json`) |
| `samples/standalone/<model>/seed_<k>/` | native structures, with `--sample` |
| `logs/<model>/seed_<k>/` | native runs without `--sample` |
| `reprs/`, `layouts/` | representations and token layouts |
| `logs/run/` | the output of every step |

## Running steps individually

Each step can run on its own in the model's environment:

```bash
D="--template-mmcif-dir examples/9mnb/mmcif --release-dates <release_date_cache.json> \
   --obsolete-pdbs <obsolete_to_successor.json> --kalign <kalign>"
I="--input examples/9mnb/9mnb.json --seed 1"

# 1. native runs (add --sample for native structures)
python scripts/run_af3.py      $I --model-dir <af3> --template-mmcif-dir examples/9mnb/mmcif
python scripts/run_protenix.py $I --mode native $D
python scripts/run_opendde.py  $I --mode native $D --checkpoint <opendde.pt>
python scripts/run_esmfold2.py $I --mode native --ckpt <esmfold2> --esmc <esmc>

# 2. token layouts
for m in af3 protenix opendde esmfold2; do python scripts/token_layout.py --pred logs/$m/seed_1 --model $m --out layouts; done

# 3. SoupFold with Protenix as anchor
python scripts/run_protenix.py $I --mode soupfold $D
```

## GPU memory

Large complexes can exceed GPU memory.

- **OpenDDE.** Its default trunk chunk sizes assume very large GPUs. On an 80 GB GPU pass smaller
  ones, e.g. `--chunk-thresholds '{"512":128,"1024":64,"1536":32,"2048":16,"3072":8,"4096":4}'`.
  The result is the same up to floating point order.
- **ESMFold2.** If 5 samples do not fit in one batch, the batch is halved automatically and the
  samples are drawn over several passes.

## Example outputs

`samples/best_conf/9mnb/` holds, for each native model and for OpenDDE+SoupFold, the top-ranked of
25 samples (seeds 1-5) with DockQ per interface in `summary.json`. Each native model docks only one
of the two Fabs. SoupFold docks both.
`samples/standalone/` and `samples/soupfold/` hold our outputs for 9y0a (seeds 1-5).

## Citation

```bibtex
@article{jang2026soupfold,
  title   = {Co-folding with a Soup of Representations},
  author  = {Jang, Hyosoon and Kim, Taewon and Ahn, Sungsoo},
  journal = {arXiv preprint arXiv:2609.15552},
  year    = {2026}
}
```
