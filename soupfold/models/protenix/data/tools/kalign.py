# Modified for SoupFold: import paths and unused parts, see soupfold/models/NOTICE.md
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kalign


class Kalign:
    """Kalign through the kalign-python package (3.4.8), in place of the Kalign binary."""

    def __init__(self, *, binary_path=None):
        pass

    def align(self, sequences):
        for s in sequences:
            if len(s) < 6:
                raise ValueError(f"Kalign requires all sequences to be at least 6 residues long. Got {s} ({len(s)} residues).")
        return kalign.align(list(sequences), seq_type="protein", n_threads=1)
