from pipeline.d2vr_pipeline import D2VRPipeline
#from diffusers import EulerAncestralDiscreteScheduler, ControlNetModel
from diffusers import ControlNetModel, UNet2DConditionModel
#from diffusers import DDPMScheduler, ControlNetModel, UNet2DConditionModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import time

def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))

# get arguments
parser = argparse.ArgumentParser(description="Test code for StableVSR.")
parser.add_argument("--out_path", default='./results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str, default='/remote-home/share/liangjianfeng/stablevsr/test_dataset/LR', help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
parser.add_argument("--controlnet_ckpt", type=str, required=True, help="Path to your folder with the controlnet checkpoint.")
parser.add_argument("--unet_ckpt", type=str, required=True, help="Path to your folder with the unet checkpoint.")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
parser.add_argument("--model_path", type=str, required=True, help="Path to the StableVSR model directory")
parser.add_argument("--scheduler", type=str, default="DDPM", help="Scheduler type to use (e.g., DPMSolverPlusPlus, EulerAncestralDiscrete)")
parser.add_argument("--of_model", type=str, default="DRFA", choices=["RAFT", "GMFlow", "FlowFormerPP", "GMA", "DRFA"], help="Optical flow model to use")
parser.add_argument("--of_path", type=str, default="./DRFA/", help="Path to the optical flow model source code directory (required for GMFlow/FlowFormerPP/GMA/SC)")
parser.add_argument("--of_pretrained", type=str, default="/remote-home/share/liangjianfeng/stablevsr/GMA-sc/results/sintel/gma/gma-degradation.pth", help="Path to the optical flow model pretrained weights")

args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

set_seed(42)
gpu_id = args.gpu_id
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_id = args.model_path

print(f"Model directory contents: {os.listdir(model_id)}")

controlnet_model = ControlNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else f"{model_id}/controlnet", local_files_only=True)
#controlnet_model = ControlNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id, subfolder='Cnet') # your own controlnet model

unet = UNet2DConditionModel.from_pretrained(args.unet_ckpt, subfolder='unet', local_files_only=True)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", local_files_only=True)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)

pipeline = D2VRPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet_model,
    scheduler=None,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

#pipeline = D2VRPipeline.from_pretrained(model_id, controlnet=controlnet_model, unet=unet)

if args.scheduler == "LMSDiscrete":
    from diffusers import LMSDiscreteScheduler
    scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler_lms', local_files_only=True)
elif args.scheduler == "EulerAncestralDiscrete":
    from diffusers import EulerAncestralDiscreteScheduler
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler_euler', local_files_only=True)
elif args.scheduler == "DDPM":
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler_ddpm', local_files_only=True)
else:
    raise ValueError(f"Unsupported scheduler type: {args.scheduler}")

pipeline.scheduler = scheduler
pipeline = pipeline.to(device)
#pipeline.enable_xformers_memory_efficient_attention()

if args.of_model == "RAFT":
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device)
    print("Using optical flow model: RAFT")
elif args.of_model == "GMFlow":
    try:
        from opticalflow.gmflow_wrapper import GMFlowWrapper
        import sys
        if args.of_path is None:
            raise ValueError("--of_path is required for GMFlow (path to gmflow source directory)")
        sys.path.insert(0, args.of_path)
        from gmflow.gmflow import GMFlow
        
        gmflow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )
        
        if args.of_pretrained is None:
            raise ValueError("--of_pretrained is required for GMFlow (path to pretrained weights)")
        checkpoint = torch.load(args.of_pretrained, map_location='cpu')
        if 'model' in checkpoint:
            gmflow_model.load_state_dict(checkpoint['model'], strict=True)
        else:
            gmflow_model.load_state_dict(checkpoint, strict=True)
        
        gmflow_model = gmflow_model.to(device)
        gmflow_model.eval()
        
        of_model = GMFlowWrapper(gmflow_model)
        of_model.requires_grad_(False)
        print("Using optical flow model: GMFlow")
    except Exception as e:
        print(f"Failed to load GMFlow: {e}")
        print("Falling back to RAFT model...")
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        of_model.requires_grad_(False)
        of_model = of_model.to(device)
        print("Using optical flow model: RAFT (fallback)")
elif args.of_model == "FlowFormerPP":
    try:
        from opticalflow.flowformer_wrapper import FlowFormerPPWrapper
        import sys
        if args.of_path is None:
            raise ValueError("--of_path is required for FlowFormerPP (path to FlowFormerPlusPlus source directory)")
        sys.path.insert(0, args.of_path)
        try:
            from core.FlowFormer import build_flowformer
            from configs.things import get_cfg as ffpp_get_cfg
        except ImportError as import_err:
            raise ImportError(f"FlowFormer++ import failed: {import_err}. This might be due to missing dependencies or version conflicts (e.g., timm version).") from import_err

        cfg = ffpp_get_cfg()
        flowformer_model = build_flowformer(cfg)
        if args.of_pretrained is None:
            raise ValueError("--of_pretrained is required for FlowFormerPP (path to pretrained weights)")
        checkpoint = torch.load(args.of_pretrained, map_location='cpu')
        state = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = flowformer_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"FlowFormer++ weights not strictly matched, missing={len(missing)}, unexpected={len(unexpected)}")

        flowformer_model = flowformer_model.to(device)
        flowformer_model.eval()

        of_model = FlowFormerPPWrapper(flowformer_model)
        of_model.requires_grad_(False)
        print("Using optical flow model: FlowFormer++")
    except Exception as e:
        error_msg = str(e)
        if 'timm' in error_msg.lower() or 'overlay_external_default_cfg' in error_msg:
            print(f"FlowFormer++ failed to load (likely due to timm version incompatibility): {error_msg}")
            print("Tip: FlowFormer++ requires a compatible timm version. Consider using a different optical flow model or updating timm.")
        else:
            print(f"Failed to load FlowFormer++: {error_msg}")
        print("Falling back to RAFT model...")
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        of_model.requires_grad_(False)
        of_model = of_model.to(device)
        print("Using optical flow model: RAFT (fallback)")
elif args.of_model == "GMA":
    try:
        from opticalflow.gma_wrapper import GMAWrapper
        import sys
        
        if args.of_path is None:
            raise ValueError("--of_path is required for GMA (path to GMA source directory)")
        gma_path = args.of_path
        gma_core_path = os.path.join(gma_path, 'core')
        for p in [gma_path, gma_core_path]:
            if p not in sys.path:
                sys.path.insert(0, p)
        
        import core.update
        import gma
        from core.network import RAFTGMA
        
        class Args:
            def __init__(self):
                self.mixed_precision = False
                self.alternate_corr = False
                self.dropout = 0.0
                self.num_heads = 1
                self.position_only = False
                self.position_and_content = True
                self.pe = 'linear'
                self.self_pe = 'linear'
                self.corr_levels = 4
                self.corr_radius = 4
            def __contains__(self, key):
                return hasattr(self, key)
        
        gma_args = Args()
        gma_model = RAFTGMA(gma_args)
        
        if args.of_pretrained is None:
            raise ValueError("--of_pretrained is required for GMA (path to pretrained weights)")
        checkpoint = torch.load(args.of_pretrained, map_location='cpu')
        if 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[len('module.'):]: v for k, v in state.items()}
        
        gma_model.load_state_dict(state, strict=True)
        gma_model = gma_model.to(device)
        gma_model.eval()
        
        of_model = GMAWrapper(gma_model)
        of_model.requires_grad_(False)
        print("Using optical flow model: GMA")
    except Exception as e:
        print(f"Failed to load GMA: {e}")
        print("Falling back to RAFT model...")
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        of_model.requires_grad_(False)
        of_model = of_model.to(device)
        print("Using optical flow model: RAFT (fallback)")
elif args.of_model == "DRFA":
    try:
        from opticalflow.gma_wrapper import GMAWrapper
        import sys
        
        if args.of_path is None:
            raise ValueError("--of_path is required for DRFA (path to GMA-SC source directory)")
        gma_path = args.of_path
        gma_core_path = os.path.join(gma_path, 'core')
        for p in [gma_path, gma_core_path]:
            if p not in sys.path:
                sys.path.insert(0, p)
        
        import core.update
        import gma
        from core.network import RAFTGMA
        
        class Args:
            def __init__(self):
                self.mixed_precision = False
                self.alternate_corr = False
                self.dropout = 0.0
                self.num_heads = 1
                self.position_only = False
                self.position_and_content = True
                self.pe = 'linear'
                self.self_pe = 'linear'
                self.corr_levels = 4
                self.corr_radius = 4
            def __contains__(self, key):
                return hasattr(self, key)
        
        gma_args = Args()
        gma_model = RAFTGMA(gma_args)
        
        if args.of_pretrained is None:
            raise ValueError("--of_pretrained is required for DRFA (path to pretrained weights)")
        checkpoint = torch.load(args.of_pretrained, map_location='cpu')
        if 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[len('module.'):]: v for k, v in state.items()}
        
        gma_model.load_state_dict(state, strict=True)
        gma_model = gma_model.to(device)
        gma_model.eval()
        
        of_model = GMAWrapper(gma_model)
        of_model.requires_grad_(False)
        print("Using optical flow model: DRFA")
    except Exception as e:
        print(f"Failed to load GMA: {e}")
        print("Falling back to RAFT model...")
        of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        of_model.requires_grad_(False)
        of_model = of_model.to(device)
        print("Using optical flow model: RAFT (fallback)")
else:
    raise ValueError(f"Unsupported optical flow model: {args.of_model}. Please choose from 'RAFT', 'GMFlow', 'FlowFormerPP', 'GMA', or 'DRFA'")

print(f"Loaded scheduler type: {type(pipeline.scheduler).__name__}")
print(f"Scheduler config: {pipeline.scheduler.config}")

assert pipeline.unet == unet, "Pipeline unet does not match expected"

# iterate for every video sequence in the input folder
seqs = sorted(os.listdir(args.in_path))
total_start_time = time.time()
for seq in seqs:
    frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))
    frames = []
    for frame_name in frame_names:
        frame = Path(os.path.join(args.in_path, seq, frame_name))
        frame = Image.open(frame)
        # frame = center_crop(frame)
        frames.append(frame)

    # upscale frames using StableVSR
    seq_start_time = time.time()
    frames = pipeline('', frames, num_inference_steps=args.num_inference_steps, guidance_scale=0, of_model=of_model, seq_name=seq).images
    seq_end_time = time.time()
    seq_inference_time = seq_end_time - seq_start_time
    print(f"Inference time for sequence {seq}: {seq_inference_time:.2f} seconds")
    frames = [frame[0] for frame in frames]
    
    # save upscaled sequences
    seq = Path(seq)
    target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
    os.makedirs(target_path, exist_ok=True)
    for frame, name in zip(frames, frame_names):
        frame.save(os.path.join(target_path, name))

total_end_time = time.time()
total_inference_time = total_end_time - total_start_time
print(f"Total inference time: {total_inference_time:.2f} seconds")
