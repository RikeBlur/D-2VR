from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pyiqa
from DISTS_pytorch import DISTS
from torchvision.models.optical_flow import raft_large as raft
import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop
from opticalflow.flow_utils import get_flow
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------------------------
# Interfaces kept for train_all.py compatibility
# ---------------------------------------------------------------------------

def init_eval_metrics(device='cuda'):
    """Initialize evaluation metric models (called from training script)."""
    lpips = LPIPS(normalize=True).to(device)
    dists = DISTS().to(device)
    psnr = PSNR(data_range=1).to(device)
    ssim = SSIM(data_range=1).to(device)
    musiq = pyiqa.create_metric('musiq', device=device, as_loss=False)
    clip = pyiqa.create_metric('clipiqa', device=device, as_loss=False)
    
    return {
        'lpips': lpips,
        'dists': dists,
        'psnr': psnr,
        'ssim': ssim,
        'musiq': musiq,
        'clip': clip
    }


def compute_metrics(gt, rec, prev_gt=None, prev_rec=None, metrics=None, of_model=None):
    """Compute per-sample evaluation metrics (called from training script).
    
    Args:
        gt: ground truth image, shape (C, H, W), range [0, 1]
        rec: reconstructed image, shape (C, H, W), range [0, 1]
        prev_gt: previous ground truth image for temporal metrics
        prev_rec: previous reconstructed image for temporal metrics
        metrics: dict of metric models
        of_model: optical flow model for temporal metrics
    
    Returns:
        dict of metric values
    """
    results = {}
    
    gt = gt.unsqueeze(0) if gt.ndim == 3 else gt
    rec = rec.unsqueeze(0) if rec.ndim == 3 else rec
    
    with torch.no_grad():
        results['psnr'] = metrics['psnr'](rec, gt).item()
        results['ssim'] = metrics['ssim'](rec, gt).item()
        results['lpips'] = metrics['lpips'](rec, gt).item()
        results['dists'] = metrics['dists'](rec, gt).item()
        results['musiq'] = metrics['musiq'](rec).item()
        results['clip'] = metrics['clip'](rec).item()
        
        if prev_gt is not None and prev_rec is not None:
            prev_gt = prev_gt.unsqueeze(0) if prev_gt.ndim == 3 else prev_gt
            prev_rec = prev_rec.unsqueeze(0) if prev_rec.ndim == 3 else prev_rec
            
            tlpips_value = (metrics['lpips'](gt, prev_gt) - metrics['lpips'](rec, prev_rec)).abs()
            results['tlpips'] = tlpips_value.item()
            
            if of_model is not None:
                tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
                results['tof'] = tof_value.item()
    
    return results


# ---------------------------------------------------------------------------
# Main: standalone evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation code for StableVSR.")
    # expected folder organization: root/sequences/frames
    parser.add_argument("--out_path", type=str, default='./StableVSR_results/',
                        help="Path to output folder containing the upscaled frames.")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to folder with GT frames. Required when --no_reference is not set.")
    parser.add_argument("--json_output", type=str, default=None,
                        help="Path to output JSON file for metrics.")
    parser.add_argument("--no_reference", action='store_true', default=False,
                        help="If set, compute no-reference metrics only (MUSIQ/MANIQA/CLIP-IQA/NIQE/DOVER). "
                             "Otherwise compute full reference metrics (PSNR/SSIM/LPIPS/DISTS/MUSIQ/MANIQA/"
                             "CLIP-IQA/NIQE/tLPIPS/tOF).")
    args = parser.parse_args()

    if not args.no_reference and args.gt_path is None:
        parser.error("--gt_path is required when --no_reference is not set.")

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    rec_path = args.out_path
    seqs = sorted(os.listdir(rec_path))
    device = torch.device('cuda')
    tt = ToTensor()

    print("\n" + "="*80)
    print("Loading evaluation models...")
    print("="*80)

    # -----------------------------------------------------------------------
    # no_reference=False: full reference metrics (eval_extended, without dover/twe)
    # -----------------------------------------------------------------------
    if not args.no_reference:
        gt_path = args.gt_path

        of_model = raft(pretrained=True).to(device)
        lpips = LPIPS(normalize=True).to(device)
        dists = DISTS().to(device)
        psnr = PSNR(data_range=1).to(device)
        ssim = SSIM(data_range=1).to(device)
        musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
        niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
        clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)
        maniqa = pyiqa.create_metric('maniqa', device='cuda', as_loss=False)
        print("✓ All models loaded successfully!\n")
        print("="*80 + "\n")

        lpips_dict = {}
        psnr_dict = {}
        ssim_dict = {}
        dists_dict = {}
        musiq_dict = {}
        niqe_dict = {}
        clip_dict = {}
        maniqa_dict = {}
        tlpips_dict = {}
        tof_dict = {}

        total = 0
        for root, dirs, files in os.walk(gt_path):
            total += len(files)
        pbar = tqdm(total=total, ncols=100)

        for seq in seqs:
            ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))
            ims_gt = sorted(os.listdir(os.path.join(gt_path, seq)))

            lpips_dict[seq] = []
            psnr_dict[seq] = []
            ssim_dict[seq] = []
            dists_dict[seq] = []
            musiq_dict[seq] = []
            niqe_dict[seq] = []
            clip_dict[seq] = []
            maniqa_dict[seq] = []
            tlpips_dict[seq] = []
            tof_dict[seq] = []

            for i, (im_rec, im_gt) in enumerate(zip(ims_rec, ims_gt)):
                with torch.no_grad():
                    gt = Image.open(os.path.join(gt_path, seq, im_gt))
                    rec = Image.open(os.path.join(rec_path, seq, im_rec))
                    gt = tt(gt).unsqueeze(0).to(device)
                    rec = tt(rec).unsqueeze(0).to(device)

                    psnr_value = psnr(gt, rec)
                    ssim_value = ssim(gt, rec)
                    lpips_value = lpips(gt, rec)
                    dists_value = dists(gt, rec)
                    musiq_value = musiq(rec)
                    niqe_value = niqe(rec)
                    clip_value = clip(rec)
                    maniqa_value = maniqa(rec)

                    if i > 0:
                        tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
                        tlpips_dict[seq].append(tlpips_value.item())
                        tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
                        tof_dict[seq].append(tof_value.item())

                psnr_dict[seq].append(psnr_value.item())
                ssim_dict[seq].append(ssim_value.item())
                lpips_dict[seq].append(lpips_value.item())
                dists_dict[seq].append(dists_value.item())
                musiq_dict[seq].append(musiq_value.item())
                niqe_dict[seq].append(niqe_value.item())
                clip_dict[seq].append(clip_value.item())
                maniqa_dict[seq].append(maniqa_value.item())

                prev_rec = rec
                prev_gt = gt
                pbar.update()

        pbar.close()

        mean_psnr = np.round(np.mean([np.mean(psnr_dict[k]) for k in psnr_dict]), 2)
        mean_ssim = np.round(np.mean([np.mean(ssim_dict[k]) for k in ssim_dict]), 3)
        mean_lpips = np.round(np.mean([np.mean(lpips_dict[k]) for k in lpips_dict]), 3)
        mean_dists = np.round(np.mean([np.mean(dists_dict[k]) for k in dists_dict]), 3)
        mean_musiq = np.round(np.mean([np.mean(musiq_dict[k]) for k in musiq_dict]), 2)
        mean_niqe = np.round(np.mean([np.mean(niqe_dict[k]) for k in niqe_dict]), 2)
        mean_clip = np.round(np.mean([np.mean(clip_dict[k]) for k in clip_dict]), 3)
        mean_maniqa = np.round(np.mean([np.mean(maniqa_dict[k]) for k in maniqa_dict]), 4)
        mean_tlpips = np.round(np.mean([np.mean(tlpips_dict[k]) for k in tlpips_dict if tlpips_dict[k]]) * 1e3, 2)
        mean_tof = np.round(np.mean([np.mean(tof_dict[k]) for k in tof_dict if tof_dict[k]]) * 1e1, 3)

        print("\n" + "="*80)
        print("EVALUATION RESULTS:")
        print("="*80)
        print(f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, '
              f'tLPIPS: {mean_tlpips}, tOF: {mean_tof}, MUSIQ: {mean_musiq}, MANIQA: {mean_maniqa}, '
              f'CLIP-IQA: {mean_clip}, NIQE: {mean_niqe}')
        print("="*80)

        if args.json_output:
            metrics_results = {
                'PSNR': float(mean_psnr),
                'SSIM': float(mean_ssim),
                'LPIPS': float(mean_lpips),
                'DISTS': float(mean_dists),
                'tLPIPS': float(mean_tlpips),
                'tOF': float(mean_tof),
                'MUSIQ': float(mean_musiq),
                'MANIQA': float(mean_maniqa),
                'CLIP-IQA': float(mean_clip),
                'NIQE': float(mean_niqe),
            }
            with open(args.json_output, 'w') as f:
                json.dump(metrics_results, f, indent=4)
            print(f'Metrics saved to {args.json_output}')

    # -----------------------------------------------------------------------
    # no_reference=True: no-reference metrics (eval_no_reference, without twe)
    # -----------------------------------------------------------------------
    else:
        musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
        niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
        clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)
        maniqa = pyiqa.create_metric('maniqa', device='cuda', as_loss=False)
        print("✓ All models loaded successfully!\n")
        print("="*80 + "\n")

        musiq_dict = {}
        niqe_dict = {}
        clip_dict = {}
        maniqa_dict = {}

        total = 0
        for root, dirs, files in os.walk(rec_path):
            total += len(files)
        pbar = tqdm(total=total, ncols=100)

        for seq in seqs:
            ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))

            musiq_dict[seq] = []
            niqe_dict[seq] = []
            clip_dict[seq] = []
            maniqa_dict[seq] = []

            for i, im_rec in enumerate(ims_rec):
                with torch.no_grad():
                    rec = Image.open(os.path.join(rec_path, seq, im_rec))
                    rec = tt(rec).unsqueeze(0).to(device)

                    musiq_value = musiq(rec)
                    niqe_value = niqe(rec)
                    clip_value = clip(rec)
                    maniqa_value = maniqa(rec)

                musiq_dict[seq].append(musiq_value.item())
                niqe_dict[seq].append(niqe_value.item())
                clip_dict[seq].append(clip_value.item())
                maniqa_dict[seq].append(maniqa_value.item())

                pbar.update()

        pbar.close()

        mean_musiq = np.round(np.mean([np.mean(musiq_dict[k]) for k in musiq_dict]), 2)
        mean_niqe = np.round(np.mean([np.mean(niqe_dict[k]) for k in niqe_dict]), 2)
        mean_clip = np.round(np.mean([np.mean(clip_dict[k]) for k in clip_dict]), 3)
        mean_maniqa = np.round(np.mean([np.mean(maniqa_dict[k]) for k in maniqa_dict]), 4)

        print("\n" + "="*80)
        print("EVALUATION RESULTS:")
        print("="*80)
        print(f'MUSIQ: {mean_musiq}, MANIQA: {mean_maniqa}, CLIP-IQA: {mean_clip}, NIQE: {mean_niqe}')
        print("="*80)

        if args.json_output:
            metrics_results = {
                'MUSIQ': float(mean_musiq),
                'MANIQA': float(mean_maniqa),
                'CLIP-IQA': float(mean_clip),
                'NIQE': float(mean_niqe),
            }
            with open(args.json_output, 'w') as f:
                json.dump(metrics_results, f, indent=4)
            print(f'Metrics saved to {args.json_output}')
