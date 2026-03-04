import os
import cv2
import numpy as np
from pathlib import Path
import random
import torch
import json

def load_degradation_params_from_json(json_path):
    """Load degradation parameters from a JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        dict: Dictionary containing degradation parameters.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print(f"Warning: JSON file not found: {json_path}, using default parameters.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: JSON file {json_path} is malformed, using default parameters.")
        return {}

def sample_parameter(param_config):
    """Sample a value from a parameter configuration.
    
    Args:
        param_config: Parameter config. Can be a scalar, a continuous range [min, max],
            or a discrete list to sample from.
            - Scalar: returned as-is, e.g. 5 or 1.5
            - Continuous range: list of two floats [min, max], e.g. [0.5, 2.0]
            - Discrete list: any other list, e.g. [1, 3] or [1, 2, 3]
        
    Returns:
        Sampled parameter value.
    """
    if isinstance(param_config, (int, float)):
        return param_config
    elif isinstance(param_config, list):
        if (len(param_config) == 2 and 
            all(isinstance(x, float) for x in param_config)):
            min_val, max_val = param_config
            return random.uniform(min_val, max_val)
        else:
            return random.choice(param_config)
    else:
        raise ValueError(f"Unsupported parameter config format: {param_config}")

def apply_degradation(img, blur_kernel_size=None, blur_sigma=None, noise_sigma=None, jpeg_quality=None, 
                     second_order=False, blur_kernel_size_2=None, blur_sigma_2=None, noise_sigma_2=None, jpeg_quality_2=None,
                     downsample_scale=None, downsample_scale_2=None, params_json_path=None):
    """Apply degradation pipeline to an image.
    
    Args:
        img: Input image (numpy array).
        blur_kernel_size: Gaussian blur kernel size for the first pass.
        blur_sigma: Gaussian blur sigma for the first pass (auto-computed if None).
        noise_sigma: Noise standard deviation for the first pass.
        jpeg_quality: JPEG compression quality for the first pass.
        second_order: Whether to apply a second-order degradation pass.
        blur_kernel_size_2: Gaussian blur kernel size for the second pass.
        blur_sigma_2: Gaussian blur sigma for the second pass (auto-computed if None).
        noise_sigma_2: Noise standard deviation for the second pass.
        jpeg_quality_2: JPEG compression quality for the second pass.
        downsample_scale: Downsampling scale factor for the first pass.
        downsample_scale_2: Downsampling scale factor for the second pass.
        params_json_path: Optional path to a JSON file containing degradation parameters.
    """
    if params_json_path:
        json_params = load_degradation_params_from_json(params_json_path)
        
        if 'blur_kernel_size' in json_params:
            blur_kernel_size = sample_parameter(json_params['blur_kernel_size'])
        if 'blur_sigma' in json_params:
            blur_sigma = sample_parameter(json_params['blur_sigma'])
        if 'noise_sigma' in json_params:
            noise_sigma = sample_parameter(json_params['noise_sigma'])
        if 'jpeg_quality' in json_params:
            jpeg_quality = sample_parameter(json_params['jpeg_quality'])
        if 'second_order' in json_params:
            second_order = json_params['second_order']
        if 'blur_kernel_size_2' in json_params:
            blur_kernel_size_2 = sample_parameter(json_params['blur_kernel_size_2'])
        if 'blur_sigma_2' in json_params:
            blur_sigma_2 = sample_parameter(json_params['blur_sigma_2'])
        if 'noise_sigma_2' in json_params:
            noise_sigma_2 = sample_parameter(json_params['noise_sigma_2'])
        if 'jpeg_quality_2' in json_params:
            jpeg_quality_2 = sample_parameter(json_params['jpeg_quality_2'])
        if 'downsample_scale' in json_params:
            downsample_scale = json_params['downsample_scale']
        if 'downsample_scale_2' in json_params:
            downsample_scale_2 = json_params['downsample_scale_2']
        
        print(f"Loaded parameters from JSON: {params_json_path}")
        print(f"Sampled params: blur1={blur_kernel_size}, sigma1={blur_sigma}, noise1={noise_sigma}, jpeg1={jpeg_quality}")
        if downsample_scale:
            print(f"Downsample scale 1: {downsample_scale}")
        if second_order:
            print(f"Sampled params: blur2={blur_kernel_size_2}, sigma2={blur_sigma_2}, noise2={noise_sigma_2}, jpeg2={jpeg_quality_2}")
            if downsample_scale_2:
                print(f"Downsample scale 2: {downsample_scale_2}")
    
    degraded = img.copy()
    
    if blur_kernel_size:
        kernel_size = int(blur_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma_x = blur_sigma if blur_sigma is not None else 0
        degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), sigma_x)
    
    if downsample_scale:
        h, w = degraded.shape[:2]
        new_h = int(h / downsample_scale)
        new_w = int(w / downsample_scale)
        degraded = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if noise_sigma:
        #noise = np.random.normal(0, noise_sigma, degraded.shape).astype(np.uint8)
        degraded = apply_realistic_noise(degraded, noise_sigma)
    
    if jpeg_quality is not None:
        # OpenCV expects an integer quality in [0, 100]
        jpeg_quality_int = int(round(float(jpeg_quality)))
        jpeg_quality_int = max(0, min(100, jpeg_quality_int))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_int]
        _, encoded = cv2.imencode('.jpg', degraded, encode_param)
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    if second_order:
        if blur_kernel_size_2:
            kernel_size_2 = int(blur_kernel_size_2)
            if kernel_size_2 % 2 == 0:
                kernel_size_2 += 1
            sigma_x_2 = blur_sigma_2 if blur_sigma_2 is not None else 0
            degraded = cv2.GaussianBlur(degraded, (kernel_size_2, kernel_size_2), sigma_x_2)
        
        if downsample_scale_2:
            h, w = degraded.shape[:2]
            new_h = int(h / downsample_scale_2)
            new_w = int(w / downsample_scale_2)
            degraded = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        if noise_sigma_2:
            #noise_2 = np.random.normal(0, noise_sigma_2, degraded.shape).astype(np.uint8)
            degraded = apply_realistic_noise(degraded, noise_sigma_2)
        
        if jpeg_quality_2 is not None:
            # OpenCV expects an integer quality in [0, 100]
            jpeg_quality_2_int = int(round(float(jpeg_quality_2)))
            jpeg_quality_2_int = max(0, min(100, jpeg_quality_2_int))
            encode_param_2 = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_2_int]
            _, encoded_2 = cv2.imencode('.jpg', degraded, encode_param_2)
            degraded = cv2.imdecode(encoded_2, cv2.IMREAD_COLOR)
        
    return degraded


def process_sequence(input_dir, output_dir_root_path, params_json_path):
    """Process an entire directory of image sequences.
    
    Args:
        input_dir: Path to the input directory containing sequence subfolders.
        output_dir_root_path: Root path for degraded output sequences.
        params_json_path: Path to the degradation parameters JSON file.
    """
    input_path_obj = Path(input_dir)
    root_output_path = Path(output_dir_root_path)
    
    root_output_path.mkdir(exist_ok=True, parents=True)
    
    for sequence_dir in input_path_obj.iterdir():
        if not sequence_dir.is_dir():
            continue
            
        print(f"Processing sequence: {sequence_dir.name}")
        
        if params_json_path:
            json_params = load_degradation_params_from_json(params_json_path)
            
            blur_kernel_size = sample_parameter(json_params.get('blur_kernel_size', 3))
            blur_sigma = sample_parameter(json_params.get('blur_sigma')) if 'blur_sigma' in json_params else None
            noise_sigma = sample_parameter(json_params.get('noise_sigma', 1.0))
            jpeg_quality = sample_parameter(json_params.get('jpeg_quality', 85))
            downsample_scale = json_params.get('downsample_scale')
            second_order = json_params.get('second_order', False)
            blur_kernel_size_2 = sample_parameter(json_params.get('blur_kernel_size_2', 5)) if second_order else None
            blur_sigma_2 = sample_parameter(json_params.get('blur_sigma_2')) if second_order and 'blur_sigma_2' in json_params else None
            noise_sigma_2 = sample_parameter(json_params.get('noise_sigma_2', 0.5)) if second_order else None
            jpeg_quality_2 = sample_parameter(json_params.get('jpeg_quality_2', 80)) if second_order else None
            downsample_scale_2 = json_params.get('downsample_scale_2') if second_order else None
            
            print(f"Degradation params for sequence {sequence_dir.name}:")
            print(f"  Pass 1: blur={blur_kernel_size}, sigma={blur_sigma}, noise={noise_sigma}, jpeg={jpeg_quality}, downsample={downsample_scale}")
            if second_order:
                print(f"  Pass 2: blur={blur_kernel_size_2}, sigma={blur_sigma_2}, noise={noise_sigma_2}, jpeg={jpeg_quality_2}, downsample={downsample_scale_2}")
        else:
            blur_kernel_size = 3
            blur_sigma = None
            noise_sigma = 1.0
            jpeg_quality = 85
            downsample_scale = None
            second_order = False
            blur_kernel_size_2 = None
            blur_sigma_2 = None
            noise_sigma_2 = None
            jpeg_quality_2 = None
            downsample_scale_2 = None
        
        sequence_specific_output_dir = root_output_path / sequence_dir.name
        sequence_specific_output_dir.mkdir(exist_ok=True)
        
        for img_file_path in sequence_dir.glob("*.png"):
            img = cv2.imread(str(img_file_path))
            if img is None:
                print(f"Cannot read image: {img_file_path}")
                continue
            
            degraded_img = apply_degradation(
                img,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma,
                noise_sigma=noise_sigma,
                jpeg_quality=jpeg_quality,
                second_order=second_order,
                blur_kernel_size_2=blur_kernel_size_2,
                blur_sigma_2=blur_sigma_2,
                noise_sigma_2=noise_sigma_2,
                jpeg_quality_2=jpeg_quality_2,
                downsample_scale=downsample_scale,
                downsample_scale_2=downsample_scale_2
            )
            
            output_image_filepath = sequence_specific_output_dir / img_file_path.name
            cv2.imwrite(str(output_image_filepath), degraded_img)
            
            print(f"Processed: {img_file_path.name}")

def apply_dynamic_degradation_batch(lq_batch, device, params_json_path):
    """Apply the same sampled random degradation to all frames in a batch.
    
    Args:
        lq_batch: Image tensor batch of shape [B, T, C, H, W] in range [-1, 1].
        device: Target device for the output tensors.
        params_json_path: Path to the degradation parameters JSON file.
    """
    b, t, c, h, w = lq_batch.shape
    
    if params_json_path:
        json_params = load_degradation_params_from_json(params_json_path)
        
        blur_kernel_size = sample_parameter(json_params.get('blur_kernel_size', 3))
        blur_sigma = sample_parameter(json_params.get('blur_sigma')) if 'blur_sigma' in json_params else None
        noise_sigma = sample_parameter(json_params.get('noise_sigma', 1.0))
        jpeg_quality = sample_parameter(json_params.get('jpeg_quality', 85))
        downsample_scale = json_params.get('downsample_scale')
        second_order = json_params.get('second_order', False)
        blur_kernel_size_2 = sample_parameter(json_params.get('blur_kernel_size_2', 5)) if second_order else None
        blur_sigma_2 = sample_parameter(json_params.get('blur_sigma_2')) if second_order and 'blur_sigma_2' in json_params else None
        noise_sigma_2 = sample_parameter(json_params.get('noise_sigma_2', 0.5)) if second_order else None
        jpeg_quality_2 = sample_parameter(json_params.get('jpeg_quality_2', 80)) if second_order else None
        downsample_scale_2 = json_params.get('downsample_scale_2') if second_order else None
    else:
        blur_kernel_size = 3
        blur_sigma = None
        noise_sigma = 1.0
        jpeg_quality = 85
        downsample_scale = None
        second_order = False
        blur_kernel_size_2 = None
        blur_sigma_2 = None
        noise_sigma_2 = None
        jpeg_quality_2 = None
        downsample_scale_2 = None
    
    degraded_batch = []
    degraded_sizes = []
    
    for i in range(b):
        frame_degraded = []
        for j in range(t):
            img_np = ((lq_batch[i,j].permute(1,2,0) + 1) * 127.5).cpu().numpy().astype(np.uint8)
            
            degraded_img = apply_degradation(
                img_np,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma,
                noise_sigma=noise_sigma,
                jpeg_quality=jpeg_quality,
                second_order=second_order,
                blur_kernel_size_2=blur_kernel_size_2,
                blur_sigma_2=blur_sigma_2,
                noise_sigma_2=noise_sigma_2,
                jpeg_quality_2=jpeg_quality_2,
                downsample_scale=downsample_scale,
                downsample_scale_2=downsample_scale_2
            )
            
            degraded_tensor = torch.from_numpy(degraded_img).permute(2,0,1).float() / 127.5 - 1
            degraded_tensor = degraded_tensor.to(device)
            frame_degraded.append(degraded_tensor)
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            _, encoded = cv2.imencode('.jpg', degraded_img, encode_param)
            degraded_sizes.append(len(encoded.tobytes()))
        
        degraded_batch.append(torch.stack(frame_degraded))
    
    return torch.stack(degraded_batch), degraded_sizes

def apply_dynamic_degradation(img_tensor, device, params_json_path):
    """Apply random degradation to a single image tensor (backward-compatible wrapper)."""
    batch_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,C,H,W]
    degraded_batch, sizes = apply_dynamic_degradation_batch(batch_tensor, device, params_json_path)
    return degraded_batch[0,0], sizes[0]

def apply_realistic_noise(img, noise_sigma):
    """Apply a simplified realistic noise model (intensity-adaptive Gaussian noise).
    
    Args:
        img: Input image (H, W, C).
        noise_sigma: Noise intensity (standard deviation).
    """
    img_float = img.astype(np.float32)
    
    base_noise = np.random.normal(0, noise_sigma, img.shape)
    
    intensity = np.mean(img_float, axis=2, keepdims=True) / 255.0
    intensity_factor = 1.0 + (1.0 - intensity) * 0.5
    adaptive_noise = base_noise * intensity_factor
    
    noisy_img = img_float + adaptive_noise
    
    noisy_img = np.clip(noisy_img, 0, 255)
    
    return noisy_img.astype(np.uint8)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply random degradation to image sequences.")
    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing image sequence subfolders.")
    parser.add_argument("--output_dir", required=True,
                       help="Path to the output directory.")
    parser.add_argument("--params_json", default='./degradation_params.json',
                       help="Path to degradation parameters JSON file (optional).")
    args = parser.parse_args()
    
    process_sequence(args.input_dir, args.output_dir, args.params_json)
