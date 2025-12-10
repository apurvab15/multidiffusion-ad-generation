"""
Prompt Refinement Module
Uses Claude API to enhance basic prompts into detailed, region-specific prompts
"""

import json
import os
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv
class PromptRefiner:
    """Refines basic prompts into detailed, region-specific prompts using Claude."""
    
    def __init__(self):
        # Load env vars for local runs and configure Gemini client.
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
    def refine_prompt(
        self,
        basic_prompt: str,
        logo_colors: Dict,
        regions: List[Dict] = None
    ) -> Dict:
        """
        Refine basic prompt into detailed prompts for each region.
        
        Args:
            basic_prompt: User's simple description
            logo_colors: Dictionary from color_extractor.analyze_logo_colors()
            regions: List of region definitions
            
        Returns:
            Dictionary with refined prompts
        """
        if regions is None:
            regions = self._get_default_regions()
        
        # Build the refinement prompt
        system_prompt = """You are an expert in advertising and visual design. 
Your task is to transform simple ad concepts into detailed, professional prompts 
for AI image generation that will create compelling advertisements."""
        
        user_prompt = f"""
I need to generate a professional advertisement with the following specifications:

**Basic Concept**: {basic_prompt}

**Brand Colors** (from logo):
- Primary: {logo_colors.get('primary_color', 'N/A')}
- Color palette: {', '.join(logo_colors.get('dominant_colors', []))}
- Style: {logo_colors.get('color_description', 'N/A')}

**Image Regions** to generate:
{self._format_regions(regions)}

Please provide:
1. **main_prompt**: A detailed main prompt for the entire ad (2-3 sentences)
2. **region_prompts**: Specific prompts for each region that maintain visual coherence
3. **color_harmony**: Strategy for using brand colors harmoniously
4. **style_guidelines**: Overall aesthetic direction

Requirements:
- Use brand colors naturally without forcing them everywhere
- Ensure each region prompt is detailed and specific
- Maintain professional advertising aesthetic
- Consider how regions will work together visually

Return your response as a JSON object with this structure:
{{
  "main_prompt": "...",
  "region_prompts": [
    {{"name": "region_name", "prompt": "detailed prompt", "importance": "high/medium/low"}}
  ],
  "color_harmony": "...",
  "style_guidelines": "..."
}}
"""
        
        response = self.model.generate_content(user_prompt)
        print(response.text)
        # Parse response
        response_text = response.text
        print(response_text)
        # Extract JSON from response
        try:
            refined = json.loads(response_text)
        except json.JSONDecodeError:
            # If Claude wraps it in markdown, extract it
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
                refined = json.loads(json_str)
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
                refined = json.loads(json_str)
            else:
                raise ValueError("Could not parse Claude's response as JSON")
        
        # Merge with user's custom region prompts if provided
        if regions:
            for i, region in enumerate(regions):
                if region.get('custom_prompt'):
                    if i < len(refined['region_prompts']):
                        refined['region_prompts'][i]['prompt'] = region['custom_prompt']
        print(refined)
        return refined
    
    def _get_default_regions(self) -> List[Dict]:
        """Default region configuration for ads."""
        return [
            {"name": "Background", "custom_prompt": ""},
            {"name": "Product/Hero Area", "custom_prompt": ""},
            {"name": "Text Overlay Zone", "custom_prompt": ""}
        ]
    
    def _format_regions(self, regions: List[Dict]) -> str:
        """Format regions for the prompt."""
        formatted = []
        for i, region in enumerate(regions, 1):
            custom = f" (User wants: {region['custom_prompt']})" if region.get('custom_prompt') else ""
            formatted.append(f"{i}. {region['name']}{custom}")
        return "\n".join(formatted)


if __name__ == "__main__":
    # Test the module
    refiner = PromptRefiner()
    
    test_prompt = "summer sale for tech gadgets"
    test_colors = {
        'primary_color': '#2563EB',
        'dominant_colors': ['#2563EB', '#3B82F6', '#60A5FA'],
        'color_description': 'vibrant blue, soft cyan'
    }
    
    result = refiner.refine_prompt(test_prompt, test_colors)
    print(json.dumps(result, indent=2))