# Modified for SoupFold: import paths and unused parts, see soupfold/models/NOTICE.md
from soupfold.models.esm.models.esmc.compatibility import ESMC, ESMCOutput
from soupfold.models.esm.models.esmc.config import EsmcConfig
from soupfold.models.esm.models.esmc.model import (
    EsmcForMaskedLM,
    EsmcForSequenceClassification,
    EsmcForTokenClassification,
    EsmcMaskedLMOutput,
    EsmcModel,
    EsmcOutput,
    EsmcPreTrainedModel,
    EsmcSequenceClassifierOutput,
    EsmcTokenClassifierOutput,
)
from soupfold.models.esm.models.esmc.sae import (
    EsmcSaeConfig,
    EsmcSaeLayer,
    EsmcSaeModel,
    EsmcSaeOutput,
    EsmcSaeParams,
)
from soupfold.models.esm.models.esmc.tokenizer import EsmcTokenizer

__all__ = [
    # Deprecated: kept so pre-rename imports keep working. See compatibility.py.
    "ESMC",
    "ESMCOutput",
    "EsmcConfig",
    "EsmcForMaskedLM",
    "EsmcForSequenceClassification",
    "EsmcForTokenClassification",
    "EsmcMaskedLMOutput",
    "EsmcModel",
    "EsmcOutput",
    "EsmcPreTrainedModel",
    "EsmcSaeConfig",
    "EsmcSaeLayer",
    "EsmcSaeModel",
    "EsmcSaeOutput",
    "EsmcSaeParams",
    "EsmcSequenceClassifierOutput",
    "EsmcTokenClassifierOutput",
    "EsmcTokenizer",
]
