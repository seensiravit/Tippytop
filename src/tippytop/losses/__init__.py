"""Model-agnostic training objectives.

The README's #1 headroom: the baseline trains pointwise logloss but is scored on
ranking metrics. These are the drop-in replacements.
"""
from .ranking import pointwise_logloss_grad, bpr_grad, listwise_softmax_grad

__all__ = ["pointwise_logloss_grad", "bpr_grad", "listwise_softmax_grad"]
