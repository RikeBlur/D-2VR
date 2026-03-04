"""GMFlow wrapper to match RAFT interface."""
import torch
import torch.nn.functional as F

class GMFlowWrapper(torch.nn.Module):
    """Wraps GMFlow to match the RAFT interface. RAFT returns [flow1, ..., final_flow]; GMFlow returns {'flow_preds': [flow]} or a tensor directly."""
    def __init__(self, gmflow_model):
        super().__init__()
        self.model = gmflow_model
        
    def forward(self, image1, image2):
        """
        Args:
            image1: (B, C, H, W), range [-1, 1] or [0, 1]
            image2: (B, C, H, W), range [-1, 1] or [0, 1]
        Returns:
            list: [flow], consistent with RAFT format, flow shape (B, 2, H, W)
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
            results = self.model(image1, image2,
                                attn_splits_list=[2],
                                corr_radius_list=[-1],
                                prop_radius_list=[-1])
        
        if isinstance(results, dict):
            flow = results['flow_preds'][-1]
        else:
            flow = results
            
        return [flow]
    
    def requires_grad_(self, requires_grad=False):
        self.model.requires_grad_(requires_grad)
        return self

