# Modified for SoupFold: import paths and unused parts, see soupfold/models/NOTICE.md
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import kalign


def resolve_kalign_binary(binary_path=None):
    return binary_path


class Kalign:
    """Kalign through the kalign-python package (3.4.8), in place of the Kalign binary."""

    def __init__(self, *, binary_path=None):
        pass

    def align(self, sequences):
        for s in sequences:
            if len(s) < 6:
                raise ValueError(f"Kalign requires all sequences to be at least 6 residues long. Got {s} ({len(s)} residues).")
        return kalign.align(list(sequences), seq_type="protein", n_threads=1)
