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

One Protenix-format JSON per system, shared by all models:

```json
[{"name": "8JT6", "sequences": [
  {"proteinChain": {"sequence": "MGCT...", "count": 1, "unpairedMsaPath": "u.a3m",
                    "pairedMsaPath": "p.a3m", "templatesPath": "hmmsearch.a3m"}},
  {"ligand": {"ligand": "CCD_EZX", "count": 1}}]}]
```

## Running

```bash
R=reprs
T="--template-mmcif-dir <mmcif> --release-dates <release_date_cache.json> \
   --obsolete-pdbs <obsolete_to_successor.json> --kalign <kalign>"

# 1. native runs (store representations)
python scripts/run_af3.py      --input 8JT6.json --seed 1 --out out/af3 --repr-root $R --model-dir <af3> --template-mmcif-dir <mmcif>
python scripts/run_protenix.py --mode native --input 8JT6.json --seed 1 --out out/protenix --repr-root $R $T
python scripts/run_opendde.py  --mode native --input 8JT6.json --seed 1 --out out/opendde --repr-root $R $T --checkpoint <opendde.pt>
python scripts/run_esmfold2.py --mode native --input 8JT6.json --seed 1 --out out/esmfold2 --repr-root $R --ckpt <esmfold2> --esmc <esmc>

# 2. token layouts
for m in af3 protenix opendde esmfold2; do python scripts/token_layout.py --pred out/$m --model $m --out layouts; done

# 3. SoupFold
python scripts/run_protenix.py --mode soupfold --input 8JT6.json --seed 1 --out out/protenix_soupfold \
    --repr-root $R --layouts layouts --teachers esmfold2,opendde,af3 $T
```

We used seeds 1-5 with 5 samples each and report the top-ranked structure by the confidence.
