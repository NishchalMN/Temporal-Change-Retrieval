import torch.nn as nn


class ExperimentalTemporalFusionModule(nn.Module):
    """
    Experimental temporal fusion module.

    The public repository preserves the model interface while the associated
    paper is under preparation.
    """

    def __init__(self, channels=768, output_dim=None):
        super().__init__()
        self.channels = channels
        self.output_dim = output_dim if output_dim is not None else channels

    def forward(self, feat_before, feat_after):
        raise NotImplementedError(
            "This experimental temporal fusion implementation is omitted "
            "pending publication. Use one of the public strategies instead: "
            "difference, concat, learned, or cross_attn."
        )
