import torch
import torch.nn as nn


class GMAWrapper(nn.Module):
    """Wraps GMA (Global Motion Aggregation) to match the RAFT interface. Returns a list whose last element is the final flow (B, 2, H, W)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image1, image2):
        """
        Args:
            image1: (B, C, H, W), range [-1,1] or [0,1]
            image2: (B, C, H, W), range [-1,1] or [0,1]
        Returns:
            list: [flow1, flow2, ..., final_flow], consistent with RAFT format, flow shape (B, 2, H, W)
        """
        if image1.min() >= -1.0 and image1.max() <= 1.0:
            if image1.min() < 0:
                # [-1, 1] -> [0, 255]
                image1 = (image1 + 1.0) * 127.5
                image2 = (image2 + 1.0) * 127.5
            else:
                # [0, 1] -> [0, 255]
                image1 = image1 * 255.0
                image2 = image2 * 255.0

        with torch.no_grad():
            flow_predictions = self.model(image1, image2, iters=12, test_mode=True)

        return flow_predictions

    def requires_grad_(self, requires_grad=False):
        self.model.requires_grad_(requires_grad)
        return self

