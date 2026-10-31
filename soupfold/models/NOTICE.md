# Upstream code in this directory

This directory holds code from three projects, not code written for SoupFold. It keeps their copyright headers and stays under
their licenses, in [`LICENSES/`](../../LICENSES). Those projects did not make, review or endorse this copy.

| directory | project | version | license |
|---|---|---|---|
| `protenix/`, `protenix_runner/`, `protenix_configs/` | [Protenix](https://github.com/bytedance/Protenix) | v2.0.0 | Apache-2.0 |
| `opendde/`, `opendde_runner/` | [OpenDDE](https://github.com/aurekaresearch/OpenDDE) | v1.1.0 | Apache-2.0 |
| `esm/` | [esm](https://github.com/Biohub/esm) | v3.4.0 | MIT |

Changes from those versions:

- Only the modules inference needs are kept. Import paths point here, and `runner` and `configs` are renamed as above.
  Files changed this way start with a line saying so.
- Parser: an entity declared as a ligand or ion stays a ligand (`protenix/data/core/parser.py`, `opendde/data/core/parser.py`), so
  a free amino acid tokenises per atom in all three models.
- `Kalign` calls the `kalign-python` package instead of a binary, and Protenix's check for that binary is removed. On 1500 query and
  template pairs, version 3.4.8 gave the same alignment as the Kalign 3.4.0 binary on 1498.
- Protenix no longer imports fair-esm, which only its ESM-embedding featuriser used. `protenix_base_default_v1.0.0` does not use it.
- `esm/data/` and the layer-norm kernels are left out. Neither is used here.
