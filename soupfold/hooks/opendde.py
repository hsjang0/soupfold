"""Trunk capture / trunk bypass for OpenDDE (1.1.0).

OpenDDE is derived from Protenix and exposes the same trunk interface
(`get_pairformer_output` -> (s_inputs, s, z), msa_module / template_embedder /
pairformer_stack / sample_diffusion), so the Protenix hooks apply unchanged.
"""
from .protenix import capture_trunk, drop_template_features, seed_diffusion, trunk_bypass  # noqa: F401
