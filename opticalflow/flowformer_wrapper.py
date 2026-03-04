import torch
import torch.nn as nn


class FlowFormerPPWrapper(nn.Module):
    """Wraps FlowFormer++ to match the RAFT/GMFlow interface. Returns a list whose last element is the final flow (B, 2, H, W)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        """
        Args:
            image1: (B, C, H, W), range [-1,1] or [0,1]
            image2: (B, C, H, W), range [-1,1] or [0,1]
        Returns:
            list: [flow], flow shape (B, 2, H, W)
        """
        if image1.min() >= -1.0 and image1.max() <= 1.0:
            if image1.min() < 0:
                image1 = (image1 + 1.0) * 127.5
                image2 = (image2 + 1.0) * 127.5
            else:
                image1 = image1 * 255.0
                image2 = image2 * 255.0

        with torch.no_grad():
            outputs = self.model(image1, image2)

        if isinstance(outputs, tuple):
            flow = outputs[0]
            if isinstance(flow, (list, tuple)):
                flow = flow[-1]
        elif isinstance(outputs, dict) and "flow_preds" in outputs:
            flow = outputs["flow_preds"][-1]
        else:
            flow = outputs

        return [flow]

    def requires_grad_(self, requires_grad=False):
        self.model.requires_grad_(requires_grad)
        return self

