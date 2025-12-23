#!/usr/bin/env python3
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform
from timm.models import create_model
import models_mamba
import os
import json
import requests
from io import BytesIO

# 设置环境变量以强制使用纯PyTorch实现
os.environ["CAUSAL_CONV1D_FORCE_FALLBACK"] = "TRUE"
os.environ["SELECTIVE_SCAN_FORCE_FALLBACK"] = "TRUE"

def get_args_parser():
    parser = argparse.ArgumentParser('Vim-Tiny ImageNet inference on Raspberry Pi', add_help=False)
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to checkpoint')
    parser.add_argument('--model', type=str, default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes in the model')
    parser.add_argument('--topk', type=int, default=5, help='Top-k predictions to show')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    return parser

def load_imagenet_classes():
    """Load ImageNet class names"""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            # Fallback to default class names
            return [f"Class {i}" for i in range(1000)]
    except:
        # Fallback to default class names
        return [f"Class {i}" for i in range(1000)]

def preprocess_image(image_path, input_size=128):
    """Preprocess the input image"""
    # Load image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    
    # Create transform
    transform = create_transform(
        input_size=input_size,
        is_training=False,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        crop_pct=0.875,
        tf_preprocessing=False
    )
    
    # Apply transform
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor

def main(args):
    print("Loading model...")
    
    try:
        # Create model
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
        )
        
        # Load checkpoint if provided
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        
        # Move model to device
        model.to(args.device)
        model.eval()
        
        # Load ImageNet class names
        print("Loading ImageNet class names...")
        class_names = load_imagenet_classes()
        
        # Preprocess image
        print(f"Preprocessing image from {args.image}")
        image_tensor = preprocess_image(args.image)
        image_tensor = image_tensor.to(args.device)
        
        # Perform inference
        print("Performing inference...")
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            topk_probs, topk_indices = torch.topk(probabilities, args.topk)
        
        # Display results
        print("\nTop predictions:")
        for i in range(args.topk):
            class_idx = topk_indices[i].item()
            prob = topk_probs[i].item()
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            print(f"{i+1}. {class_name}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nPlease make sure you have set the environment variables:")
        print("export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE")
        print("export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)