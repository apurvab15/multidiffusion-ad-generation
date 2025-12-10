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
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
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
    
    # Run generation
    results = pipeline.generate(
        basic_prompt=args.prompt,
        logo_path=args.logo,
        regions=regions,
        output_name=args.output,
        seed=args.seed,
        save_intermediate=not args.no_save_intermediate,
        optimization_level=args.optimization_level
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
                optimization_level=args.optimization_level
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