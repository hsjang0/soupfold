# SoupFold

SoupFold folds a complex with a base co-folding model after averaging its trunk pair
representation with those of the other models, each mapped into the base's space by a learned map.

```
z' = denorm_b( (norm_b(z_b) + sum_t f_{t->b}(norm_t(z_t))) / (1 + T) )
```

The maps are in `weights/`.

## Models

| model | version | role |
|---|---|---|
| AlphaFold3 | v3.0.1 | teacher |
| Protenix | 2.0.0 | base, teacher |
| OpenDDE | 1.1.0 | base, teacher |
| ESMFold2 | esm 3.4.0 | base, teacher |

Patches in `patches/` make the models tokenise inputs identically.
Representations are captured and injected at runtime by `soupfold/hooks/`, without editing the installed packages.

## Input

One Protenix-format JSON per system, shared by all models. Paths are relative to the JSON file.
`examples/9y0a/` is a ready-to-run ARK-AB example (antibody Fab with its antigen).

```json
[{"name": "9y0a", "sequences": [
  {"proteinChain": {"sequence": "MGVF...", "count": 1, "unpairedMsaPath": "msa/chain1_unpaired.a3m",
                    "pairedMsaPath": "msa/chain1_paired.a3m", "templatesPath": "templates/chain1_hmmsearch.a3m"}},
  {"ligand": {"ligand": "CCD_NAG", "count": 2}}]}]
```

## Pipeline

Copy `config.example.json` to `config.json` and fill in each model's Python environment and weights.

```bash
python scripts/soupfold.py --config config.json --input examples/9y0a/9y0a.json
```

It runs the four models natively (storing their representations), writes token layouts, then runs
SoupFold with Protenix, OpenDDE and ESMFold2 as base. Outputs go under `--workdir` (default `soupfold_out/`):

| directory | content |
|---|---|
| `samples/soupfold/<base>/seed_<k>/` | SoupFold structures and confidences |
| `samples/standalone/<model>/seed_<k>/` | native structures, with `--sample` (5 samples, default steps) |
| `logs/<model>/seed_<k>/` | native runs without `--sample` (1 sample, 2 steps) |
| `reprs/`, `layouts/`, `logs/run/` | representations, token layouts, step logs |

`samples/` holds our outputs for `examples/9y0a` (seeds 1-5) in the same layout.

`--seeds 1-5 --sample` reproduces the paper setting. We report the top-ranked structure by the base
model's confidence.

## Individual scripts

Each step can also be run in the model's own environment:

```bash
D="--template-mmcif-dir examples/9y0a/mmcif --release-dates <release_date_cache.json> \
   --obsolete-pdbs <obsolete_to_successor.json> --kalign <kalign>"
I="--input examples/9y0a/9y0a.json --seed 1"

python scripts/run_af3.py      $I --model-dir <af3> --template-mmcif-dir examples/9y0a/mmcif
python scripts/run_protenix.py $I --mode native $D
python scripts/run_opendde.py  $I --mode native $D --checkpoint <opendde.pt>
python scripts/run_esmfold2.py $I --mode native --ckpt <esmfold2> --esmc <esmc>

for m in af3 protenix opendde esmfold2; do python scripts/token_layout.py --pred logs/$m/seed_1 --model $m --out layouts; done

python scripts/run_protenix.py $I --mode soupfold $D
```

Add `--sample` to a native run to sample real structures into `samples/standalone/`.
