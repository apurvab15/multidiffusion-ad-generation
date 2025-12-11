#!/usr/bin/env python3
"""
Ad Generation Tool - Main Entry Point

Usage:
    python main.py --prompt "summer sale" --logo logo.png
    python main.py --prompt "tech product launch" --seed 42
    python main.py --batch prompts.txt --logo logo.png
"""

import argparse
import sys
from pathlib import Path
from backend.pipeline import AdGenerationPipeline
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Ad Generation with MultiDiffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --prompt "summer sale for laptops" --logo brand.png
  python main.py --prompt "luxury watch ad" --seed 42 --optimization-level high
  python main.py --batch test_prompts.txt --no-optimization
  python main.py --prompt "tech ad" --bbox "100,100,400,400" --region-prompt "product shot"
  python main.py --prompt "product ad" --logo logo.png --mask masks/mask1.png --mask-prompt "product showcase"
  python main.py --prompt "ad" --masks mask1.png mask2.png --mask-prompts "product" "background"
  python main.py --prompt "summer sale" --negative-prompt "blurry, low quality, distorted, text"
        """
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Basic ad prompt (e.g., "summer sale for tech products")'
    )
    
    parser.add_argument(
        '--logo', '-l',
        type=str,
        help='Path to brand logo image'
    )
    
    parser.add_argument(
        '--regions', '-r',
        type=str,
        help='Path to JSON file with custom region definitions'
    )
    
    parser.add_argument(
        '--bbox',
        action='append',
        help='Bounding box for region: x1,y1,x2,y2. Can specify multiple times.'
    )
    
    parser.add_argument(
        '--region-prompt',
        action='append',
        help='Custom prompt for each bbox (in same order). Optional.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output name (default: auto-generated timestamp)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to text file with multiple prompts (one per line)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (default: cuda). Use "cuda:0", "cuda:1", etc. for specific GPUs, or "cpu"'
    )
    
    parser.add_argument(
        '--no-optimization',
        action='store_true',
        help='Disable perceptual optimization'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high'],
        help='Perceptual optimization intensity (default: medium)'
    )
    
    parser.add_argument(
        '--no-save-intermediate',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        help='Path to single mask image file (grayscale, white=region)'
    )
    
    parser.add_argument(
        '--mask-prompt',
        type=str,
        help='Prompt for the single masked region (used with --mask)'
    )
    
    parser.add_argument(
        '--masks',
        nargs='+',
        help='Paths to multiple mask image files (for 2+ masks)'
    )
    
    parser.add_argument(
        '--mask-prompts',
        nargs='+',
        help='Prompts for each mask (must match number of --masks)'
    )
    
    parser.add_argument(
        '--mask-feather',
        type=int,
        default=10,
        help='Feathering amount for mask edges (default: 10)'
    )
    
    parser.add_argument(
        '--logo-mask',
        type=str,
        help='Mask image to determine logo position (logo will be placed at mask centroid)'
    )
    
    parser.add_argument(
        '--background-prompt',
        type=str,
        help='Specific prompt for the background'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (faster generation)'
    )
    
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip perceptual optimization step (faster generation)'
    )
    
    parser.add_argument(
        '--negative-prompt', '-n',
        type=str,
        help='Negative prompt (things to avoid in the image, e.g., "blurry, low quality, distorted")'
    )
    
    return parser.parse_args()


def parse_bbox(bbox_str):
    """Parse bbox string like '100,100,400,400' to tuple."""
    try:
        coords = [int(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError
        return tuple(coords)
    except:
        print(f"Error: Invalid bbox format '{bbox_str}'. Use: x1,y1,x2,y2")
        sys.exit(1)


def load_regions(regions_path: str) -> list:
    """Load custom region definitions from JSON file."""
    with open(regions_path, 'r') as f:
        regions = json.load(f)
    return regions


def run_single_generation(pipeline, args):
    """Run single ad generation."""
    if not args.prompt:
        print("Error: --prompt is required for single generation")
        sys.exit(1)
    
    # Load custom regions if provided
    regions = None
    if args.regions:
        regions = load_regions(args.regions)
    elif args.bbox:
        # Build regions from bbox arguments
        regions = []
        for i, bbox_str in enumerate(args.bbox):
            bbox = parse_bbox(bbox_str)
            custom_prompt = ""
            if args.region_prompt and i < len(args.region_prompt):
                custom_prompt = args.region_prompt[i]
            
            regions.append({
                "name": f"Region {i+1}",
                "bbox": bbox,
                "custom_prompt": custom_prompt,
                "importance": "high"
            })
    
    # Validate mask arguments
    if args.mask and args.masks:
        print("Error: Cannot use both --mask and --masks. Use --masks for multiple masks.")
        sys.exit(1)
    
    if args.masks and args.mask_prompts:
        if len(args.mask_prompts) != len(args.masks):
            print(f"Error: Number of --mask-prompts ({len(args.mask_prompts)}) must match number of --masks ({len(args.masks)})")
            sys.exit(1)
    
    # Run generation
    results = pipeline.generate(
        basic_prompt=args.prompt,
        logo_path=args.logo,
        regions=regions,
        output_name=args.output,
        seed=args.seed,
        save_intermediate=not args.no_save_intermediate,
        optimization_level=args.optimization_level,
        mask_path=args.mask,
        mask_paths=args.masks,
        mask_prompt=args.mask_prompt,
        mask_prompts=args.mask_prompts,
        mask_feather=args.mask_feather,
        logo_mask_path=args.logo_mask,
        background_prompt=args.background_prompt,
        skip_evaluation=args.skip_evaluation,
        skip_optimization=args.skip_optimization,
        negative_prompt=args.negative_prompt
    )
    
    return results


def run_batch_generation(pipeline, args):
    """Run batch generation from file."""
    if not args.batch:
        print("Error: --batch file is required")
        sys.exit(1)
    
    # Load prompts
    with open(args.batch, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"BATCH GENERATION: {len(prompts)} prompts")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Batch {i}/{len(prompts)}] Processing: {prompt}")
        
        output_name = f"batch_{i:03d}" if not args.output else f"{args.output}_{i:03d}"
        
        try:
            results = pipeline.generate(
                basic_prompt=prompt,
                logo_path=args.logo,
                output_name=output_name,
                seed=args.seed + i if args.seed else None,
                save_intermediate=not args.no_save_intermediate,
                optimization_level=args.optimization_level,
                skip_evaluation=args.skip_evaluation,
                skip_optimization=args.skip_optimization,
                negative_prompt=args.negative_prompt
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            continue
    
    # Create batch summary
    print(f"\n{'='*60}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully generated: {len(all_results)}/{len(prompts)} ads")
    
    # Calculate average metrics
    if all_results:
        avg_metrics = {}
        for metric in all_results[0]['metrics'].keys():
            avg_metrics[metric] = sum(r['metrics'][metric] for r in all_results) / len(all_results)
        
        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    return all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║   AI Ad Generation System                            ║
    ║   MultiDiffusion + Prompt Engineering                ║
    ║   + Perceptual Optimization                          ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    try:
        pipeline = AdGenerationPipeline(
            device=args.device,
            enable_optimization=not args.no_optimization
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have a CUDA-capable GPU")
        print("  2. Check that PyTorch is installed correctly")
        print("  3. Try running with --device cpu (slower)")
        print("  4. Use --device cuda:0, cuda:1, etc. for specific GPUs")
        sys.exit(1)
    
    # Run generation
    try:
        if args.batch:
            results = run_batch_generation(pipeline, args)
        else:
            results = run_single_generation(pipeline, args)
        
        print("\n✓ All done! Check the outputs/ directory for results.\n")
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()