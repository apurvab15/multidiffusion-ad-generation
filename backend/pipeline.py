"""
Ad Generation Pipeline
Orchestrates all modules for end-to-end ad generation
"""

import os
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from backend.color_extractor import analyze_logo_colors
from backend.prompt_refiner import PromptRefiner
from backend.multidiffuser import MultiDiffusionGenerator
from backend.evaluator import AdEvaluator
from backend.perceptual_optimizer import PerceptualOptimizer
from config import OUTPUT_DIR


class AdGenerationPipeline:
    """End-to-end pipeline for ad generation with perceptual optimization."""
    
    def __init__(self, device: str = "cuda", enable_optimization: bool = True):
        print("Initializing Ad Generation Pipeline...")
        
        self.prompt_refiner = PromptRefiner()
        self.generator = MultiDiffusionGenerator(device=device)
        #self.evaluator = AdEvaluator(device=device)
        
        self.enable_optimization = enable_optimization
        if enable_optimization:
            self.optimizer = PerceptualOptimizer(device=device)
            print("  ✓ Perceptual optimization enabled")
        else:
            self.optimizer = None
            print("  ⚠ Perceptual optimization disabled")
        
        print("Pipeline ready!")
    
    def generate(
        self,
        basic_prompt: str,
        logo_path: Optional[str] = None,
        regions: Optional[List[Dict]] = None,
        output_name: Optional[str] = None,
        seed: Optional[int] = None,
        save_intermediate: bool = True,
        optimization_level: str = "medium",
        mask_path: Optional[str] = None,
        mask_paths: Optional[List[str]] = None,
        mask_prompt: Optional[str] = None,
        mask_prompts: Optional[List[str]] = None,
        mask_feather: int = 10,
        logo_position: Optional[Tuple[int, int]] = None,
        logo_mask_path: Optional[str] = None,
        logo_size: Optional[Tuple[int, int]] = None,
        preserve_logo_aspect: bool = True,
        logo_blend_mode: str = "soft_light",
        logo_opacity: float = 0.9,
        logo_feather: bool = True,
        logo_feather_radius: int = 5,
        background_prompt: Optional[str] = None,
        skip_evaluation: bool = False,
        skip_optimization: bool = False,
        negative_prompt: Optional[str] = None
    ) -> Dict:
        """Generate ad with perceptual optimization."""
        
        print("\n" + "="*60)
        if self.enable_optimization:
            print("STARTING AD GENERATION WITH PERCEPTUAL OPTIMIZATION")
        else:
            print("STARTING AD GENERATION (OPTIMIZATION DISABLED)")
        print("="*60)
        
        # Create output directory
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"ad_{timestamp}"
        
        output_dir = OUTPUT_DIR / output_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {
            'basic_prompt': basic_prompt,
            'output_dir': str(output_dir),
            'seed': seed
        }
        
        # Step 1: Analyze logo colors
        logo_image = None
        logo_colors = None
        logo_analysis = None
        
        if logo_path:
            if os.path.exists(logo_path):
                print("\n[1/6] Analyzing logo colors...")
                logo_analysis = analyze_logo_colors(logo_path)
                logo_colors = logo_analysis
                logo_image = Image.open(logo_path)
                
                print(f"  ✓ Logo loaded: {logo_path}")
                print(f"  ✓ Logo size: {logo_image.size}")
                print(f"  ✓ Primary color: {logo_analysis['primary_color']}")
                print(f"  ✓ Color palette: {logo_analysis['color_description']}")
                
                results['logo_analysis'] = logo_analysis
                
                if save_intermediate:
                    with open(output_dir / "logo_analysis.json", 'w') as f:
                        json.dump(logo_analysis, f, indent=2)
            else:
                print(f"\n[1/6] WARNING: Logo file not found: {logo_path}")
                print("  Continuing without logo...")
                logo_image = None
                logo_colors = {
                    'primary_color': '#000000',
                    'dominant_colors': ['#000000'],
                    'color_description': 'neutral colors'
                }
        else:
            print("\n[1/6] No logo provided, skipping color analysis")
            logo_image = None
            logo_colors = {
                'primary_color': '#000000',
                'dominant_colors': ['#000000'],
                'color_description': 'neutral colors'
            }
        
        # Step 2: Refine prompts
        print("\n[2/6] Refining prompts with AI...")
        refined_prompts = self.prompt_refiner.refine_prompt(
            basic_prompt,
            logo_colors,
            regions
        )
        
        print(f"  ✓ Main prompt: {refined_prompts['main_prompt'][:80]}...")
        print(f"  ✓ Generated {len(refined_prompts['region_prompts'])} region prompts")
        
        results['refined_prompts'] = refined_prompts
        
        if save_intermediate:
            with open(output_dir / "refined_prompts.json", 'w') as f:
                json.dump(refined_prompts, f, indent=2)
        
        # Step 3: Generate ad
        print("\n[3/6] Generating advertisement...")
        print("  This may take 1-2 minutes depending on your GPU...")
        
        # Use default logo position if not specified
        if logo_position is None:
            logo_position = (600, 50)
        
        # Debug: Check if logo will be added
        if logo_image:
            print(f"  ✓ Logo will be added at position: {logo_position}")
            if logo_mask_path:
                print(f"  ✓ Using logo mask for positioning: {logo_mask_path}")
        else:
            print("  ⚠ No logo image to add")
        
        # Use default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, ugly, deformed"
        
        print(f"  ✓ Negative prompt: '{negative_prompt[:100]}...'")
        
        generated_image = self.generator.generate_ad(
            main_prompt=refined_prompts['main_prompt'],
            region_prompts=refined_prompts['region_prompts'],
            logo_image=logo_image,
            logo_position=logo_position,  # Manual position (x, y)
            logo_mask_path=logo_mask_path,  # Or use mask to determine position
            logo_size=logo_size,  # None = use logo as-is (original size)
            preserve_logo_aspect=preserve_logo_aspect,  # Preserve aspect ratio if resizing needed
            logo_blend_mode=logo_blend_mode,  # Blending mode for natural integration
            logo_opacity=logo_opacity,  # Opacity (0.0-1.0) for blending
            logo_feather=logo_feather,  # Soften edges for better blending
            logo_feather_radius=logo_feather_radius,  # Edge feathering radius
            seed=seed,
            negative_prompt=negative_prompt,
            mask_path=mask_path,
            mask_paths=mask_paths,
            mask_prompt=mask_prompt,
            mask_prompts=mask_prompts,
            mask_feather=mask_feather,
            background_prompt=background_prompt  # Optional specific background prompt
        )
        
        print("  ✓ Initial image generated successfully!")
        
        # Save initial image
        if save_intermediate:
            initial_path = output_dir / "generated_initial.png"
            generated_image.save(initial_path)
            print(f"  ✓ Initial image saved")
        
        # Step 4: Perceptual Optimization
        optimization_results = None
        final_image = generated_image
        
        if self.enable_optimization and self.optimizer and not skip_optimization:
            print("\n[4/6] Applying perceptual optimization...")
            
            target_regions = self._extract_target_regions(
                refined_prompts['region_prompts']
            )
            
            focus_keywords = self._extract_focus_keywords(
                basic_prompt,
                refined_prompts
            )
            
            optimization_results = self.optimizer.optimize(
                image=generated_image,
                target_regions=target_regions,
                focus_keywords=focus_keywords,
                enhancement_level=optimization_level
            )
            
            final_image = optimization_results['optimized']
            
            print("  ✓ Optimization complete!")
            
            if save_intermediate:
                saliency_viz = self._visualize_saliency(
                    generated_image,
                    optimization_results['saliency_map']
                )
                saliency_viz.save(output_dir / "saliency_heatmap.png")
                
                with open(output_dir / "optimization_analysis.json", 'w') as f:
                    json.dump(optimization_results['analysis'], f, indent=2)
        
        else:
            print("\n[4/6] Perceptual optimization disabled, skipping...")
        
        # Save final image
        image_path = output_dir / "generated_ad.png"
        final_image.save(image_path)
        results['image_path'] = str(image_path)
        results['optimization_results'] = optimization_results
        
        # Step 5: Evaluate
        if skip_evaluation:
            print("\n[5/6] Evaluation skipped")
            results['metrics'] = {}
        else:
            print("\n[5/6] Evaluating advertisement...")
            
            eval_metrics = self.evaluator.evaluate_ad(
                image=final_image,
                prompt=refined_prompts['main_prompt'],
                logo_colors=logo_analysis['dominant_colors'] if logo_analysis else None,
                original_logo=logo_image,
                logo_bbox=(600, 50, 600 + 128, 50 + 128) if logo_image else None
            )
            
            if optimization_results:
                eval_metrics['perceptual_quality'] = (
                    optimization_results['analysis']['contrast_improvement'] +
                    optimization_results['analysis']['sharpness_improvement']
                ) / 2
                eval_metrics['saliency_alignment'] = optimization_results['analysis']['saliency_score']
            
            print("  Evaluation Results:")
            for metric, value in eval_metrics.items():
                print(f"    • {metric}: {value:.3f}")
            
            results['metrics'] = eval_metrics
            
            if save_intermediate:
                with open(output_dir / "metrics.json", 'w') as f:
                    json.dump(eval_metrics, f, indent=2)
        
        # Step 6: Create report
        print("\n[6/6] Creating summary report...")
        self._create_report(results, output_dir)
        
        print("\n" + "="*60)
        print(f"✓ GENERATION COMPLETE!")
        print(f"  Output saved to: {output_dir}")
        if optimization_results:
            print(f"  Perceptual quality improved by: {eval_metrics.get('perceptual_quality', 0):.1%}")
        print("="*60 + "\n")
        
        return results
    
    def _extract_target_regions(self, region_prompts: List[Dict]) -> List[tuple]:
        """Extract bounding boxes of important regions."""
        target_regions = []
        
        for region in region_prompts:
            if region.get('importance') in ['critical', 'high']:
                if 'bbox' in region and region['bbox']:
                    target_regions.append(region['bbox'])
        
        return target_regions
    
    def _extract_focus_keywords(
        self,
        basic_prompt: str,
        refined_prompts: Dict
    ) -> str:
        """Extract keywords for saliency detection."""
        keywords = set()
        
        product_words = ['product', 'laptop', 'watch', 'phone', 'car', 
                        'bottle', 'shoes', 'bag', 'device', 'gadget',
                        'tech', 'electronics', 'skincare', 'cosmetic']
        
        for word in basic_prompt.lower().split():
            if any(pw in word for pw in product_words):
                keywords.add(word)
        
        keywords.update(['product', 'logo', 'brand'])
        
        return ', '.join(list(keywords)[:6])
    
    def _visualize_saliency(
        self,
        image: Image.Image,
        saliency_map: np.ndarray
    ) -> Image.Image:
        """Create visualization of saliency map."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Ad', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(saliency_map, cmap='hot')
        axes[1].set_title('Saliency Map\n(Red = High Attention)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(image)
        axes[2].imshow(saliency_map, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        viz_image = Image.open(buf).copy()
        plt.close()
        
        return viz_image
    
    def _create_report(self, results: Dict, output_dir: Path):
        """Create enhanced HTML report with optimization metrics."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ad Generation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2563EB;
            border-bottom: 4px solid #2563EB;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 40px 0;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-name {{
            font-size: 13px;
            text-transform: uppercase;
            opacity: 0.9;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin-top: 8px;
        }}
        .prompt-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            border-left: 5px solid #10b981;
        }}
        .region-prompt {{
            background: white;
            padding: 15px;
            margin: 12px 0;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .colors {{
            display: flex;
            gap: 12px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .color-box {{
            width: 70px;
            height: 70px;
            border-radius: 10px;
            border: 3px solid #fff;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .optimization-section {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 30px;
            border-radius: 12px;
            margin: 30px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge-critical {{ background: #ef4444; color: white; }}
        .badge-high {{ background: #f59e0b; color: white; }}
        .badge-medium {{ background: #3b82f6; color: white; }}
        .info-box {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Ad Generation Report</h1>
        <p style="color: #666; font-size: 14px;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Generated Advertisement</h2>
            <div class="image-container">
                <img src="generated_ad.png" alt="Generated Ad">
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Evaluation Metrics</h2>
            <div class="metrics">
"""
        
        # Add all metrics
        metric_colors = [
            'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
            'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
            'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
        ]
        
        for i, (metric, value) in enumerate(results['metrics'].items()):
            percentage = int(value * 100)
            metric_display = metric.replace('_', ' ').title()
            color = metric_colors[i % len(metric_colors)]
            
            html_content += f"""
                <div class="metric-card" style="background: {color};">
                    <div class="metric-name">{metric_display}</div>
                    <div class="metric-value">{percentage}%</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
        
        # Add optimization section if available
        if results.get('optimization_results'):
            opt = results['optimization_results']['analysis']
            html_content += f"""
        <div class="optimization-section">
            <h2 style="margin-top: 0;">✨ Perceptual Optimization Results</h2>
            
            <div class="metrics" style="margin-top: 20px;">
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="metric-name">Saliency Score</div>
                    <div class="metric-value">{opt['saliency_score']:.3f}</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="metric-name">Contrast Boost</div>
                    <div class="metric-value">+{opt['contrast_improvement']:.1%}</div>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <div class="metric-name">Sharpness Boost</div>
                    <div class="metric-value">+{opt['sharpness_improvement']:.1%}</div>
                </div>
            </div>
            
            <h3>Saliency Analysis</h3>
            <img src="saliency_heatmap.png" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
            
            <div class="info-box">
                <strong>📍 What These Metrics Mean:</strong>
                <ul style="margin: 10px 0;">
                    <li><strong>Saliency Score:</strong> Measures how well visual attention is directed to key elements (products, logos). Higher is better.</li>
                    <li><strong>Contrast Boost:</strong> Improvement in local contrast, making important elements stand out more.</li>
                    <li><strong>Sharpness Boost:</strong> Enhanced edge definition in high-attention regions for crisper visuals.</li>
                </ul>
            </div>
        </div>
"""
        
        # Add prompts section
        html_content += f"""
        <div class="section">
            <h2>💡 Input Prompt</h2>
            <div class="prompt-box">
                <strong>Basic Prompt:</strong><br>
                {results['basic_prompt']}
            </div>
        </div>
"""
        
        # Add refined prompts
        if 'refined_prompts' in results:
            html_content += f"""
        <div class="section">
            <h2>✨ AI-Refined Prompts</h2>
            <div class="prompt-box">
                <strong>Main Prompt:</strong><br>
                {results['refined_prompts']['main_prompt']}
            </div>
            
            <h3 style="margin-top: 30px;">Region-Specific Prompts:</h3>
"""
            for region in results['refined_prompts']['region_prompts']:
                importance = region.get('importance', 'medium')
                badge_class = f"badge-{importance}"
                html_content += f"""
            <div class="region-prompt">
                <strong>{region['name']}</strong>
                <span class="badge {badge_class}">{importance.upper()}</span>
                <p style="margin: 10px 0 0 0; color: #666;">{region['prompt']}</p>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Add logo colors
        if 'logo_analysis' in results:
            html_content += f"""
        <div class="section">
            <h2>🎨 Brand Colors</h2>
            <div class="colors">
"""
            for color in results['logo_analysis']['dominant_colors']:
                html_content += f"""
                <div class="color-box" style="background-color: {color};" title="{color}"></div>
"""
            html_content += f"""
            </div>
            <p><strong>Color Description:</strong> {results['logo_analysis']['color_description']}</p>
        </div>
"""
        
        # Footer
        html_content += f"""
        <div class="section" style="border-top: 2px solid #e5e7eb; padding-top: 30px; margin-top: 50px;">
            <p style="color: #999; text-align: center;">
                <strong>🚀 Generated with:</strong><br>
                MultiDiffusion + AI Prompt Refinement + Perceptual Optimization<br>
                <small>Anthropic Claude API • Stable Diffusion • Grounding DINO</small>
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_dir / "report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  ✓ Enhanced report saved: {output_dir / 'report.html'}")


if __name__ == "__main__":
    # Test pipeline
    pipeline = AdGenerationPipeline(enable_optimization=True)
    
    results = pipeline.generate(
        basic_prompt="summer sale for laptops",
        logo_path=None,
        seed=42,
        optimization_level="medium"
    )
    
    print("\nResults:", results)