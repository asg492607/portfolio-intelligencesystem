import os
import re
import json
import uuid
import datetime
import httpx
from bs4 import BeautifulSoup

# PyMuPDF is fitz, fallback to pypdf if fitz not installed
try:
    import fitz
    use_fitz = True
except ImportError:
    use_fitz = False
    from pypdf import PdfReader

# Default weights profile for calculation
DEFAULT_WEIGHTS = {
    "deep_analysis": 0.15,
    "skill_extractor": 0.15,
    "innovation_score": 0.10,
    "benchmarking": 0.10,
    "hidden_talent": 0.10,
    "portfolio_coach": 0.15,
    "tech_depth": 0.15,
    "consistency": 0.10
}

TECH_KEYWORDS = {
    "design_tool": ["figma", "sketch", "photoshop", "illustrator", "adobe xd", "invision", "miro", "canva", "zeplin", "framer"],
    "dev_tool": ["react", "vue", "angular", "next.js", "node.js", "javascript", "typescript", "python", "django", "fastapi", "flask", "postgresql", "mongodb", "docker", "aws", "git", "tailwind", "css", "html"],
    "methodology": ["user research", "wireframing", "prototyping", "usability testing", "agile", "scrum", "design thinking", "information architecture", "persona", "user flows"],
    "soft_skill": ["communication", "collaboration", "leadership", "problem solving", "time management", "adaptability", "critical thinking"]
}

# 1. PDF parsing using PyMuPDF (fitz)
def extract_text_from_pdf(file_path: str) -> str:
    text_content = ""
    try:
        if use_fitz:
            doc = fitz.open(file_path)
            for page in doc:
                text_content += page.get_text() + "\n"
        else:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text_content.strip()

# 2. Web Scraping for Website / Behance
async def scrape_url_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Strip scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Extract text contents
                lines = (line.strip() for line in soup.get_text().splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                title = soup.title.string if soup.title else "Scraped Portfolio"
                return f"URL: {url}\nTitle: {title}\nContent:\n{text[:15000]}"
            else:
                return f"Failed to retrieve URL {url}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error occurred scraping URL {url}: {str(e)}"

# 3. Figma API Parser
def extract_figma_file_key(url: str) -> str:
    """Extract 22-character alpha-numeric Figma key from URL."""
    match = re.search(r'/(?:file|design)/([a-zA-Z0-9]{22,})', url)
    return match.group(1) if match else None

async def parse_figma_content(url: str) -> str:
    file_key = extract_figma_file_key(url)
    if not file_key:
        return f"Figma URL: {url}\nError: Could not extract Figma File Key from the URL."

    token = os.getenv("FIGMA_ACCESS_TOKEN")
    if not token:
        # Graceful fallback: simulate file layout information
        return f"Figma URL: {url}\nFile Key: {file_key}\nWarning: FIGMA_ACCESS_TOKEN is missing. Fallback to mock Figma structural parsing.\nNodes: [Text Node: 'Hero Header', Frame Node: 'Mockup Screen Desktop', Text Node: 'Portfolio Projects', Component Node: 'Card component', Style: 'Inter 14px Regular', Colors: ['#0f172a', '#38bdf8', '#ffffff']]"

    try:
        headers = {"X-Figma-Token": token}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"https://api.figma.com/v1/files/{file_key}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                text_layers = []
                styles_count = len(data.get("styles", {}))
                components_count = len(data.get("components", {}))
                
                # Helper to traverse Figma JSON Document Nodes
                def traverse_nodes(node):
                    if node.get("type") == "TEXT":
                        char_text = node.get("characters", "").strip()
                        if char_text:
                            text_layers.append(char_text)
                    if "children" in node:
                        for child in node["children"]:
                            traverse_nodes(child)

                if "document" in data:
                    traverse_nodes(data["document"])

                compiled_text = "\n".join(text_layers[:400]) # Cap to avoid huge string sizes
                return f"Figma File: {data.get('name', 'Unnamed')}\nComponents count: {components_count}\nStyles count: {styles_count}\nContent:\n{compiled_text}"
            else:
                return f"Figma API error. Status: {response.status_code}. Detail: {response.text}"
    except Exception as e:
        return f"Error calling Figma API: {str(e)}"

# 4. Generate Embeddings (Optional Vector representation)
async def generate_text_embedding(text: str) -> list:
    """Generate embedding vector using Gemini Embedding API (or return dummy vector)."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text[:2000],
                task_type="retrieval_document"
            )
            if "embedding" in result:
                return result["embedding"]
        except Exception as e:
            print(f"Error generating embedding via Gemini API: {e}")
            
    # Dummy embedding fallback matching Qdrant size
    import random
    return [random.uniform(-0.1, 0.1) for _ in range(768)]

# Heuristics local engine fallback (already built, import same)
from analyzer_heuristics import run_heuristic_analysis

def run_ai_analysis(text: str, filename: str, role_target: str, seniority: str) -> dict:
    """Runs analysis using Gemini, OpenAI, or falls back to heuristics."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    prompt = f"""
    You are Portfolio Intelligence Agent — an expert AI system that evaluates design and development portfolios like a senior UX lead, creative director, and technical recruiter combined.

    Evaluate this candidate's portfolio.
    Source Context: {filename}
    Target Role: {role_target}
    Target Seniority: {seniority}
    
    Portfolio text content extracted:
    ---
    {text[:9000]}
    ---

    Please return a JSON matching EXACTLY this unified structure. Do not output markdown, preambles, or explanations. Just valid, parsable JSON matching this template:
    {{
      "report_id": "unique-uuid",
      "candidate_id": "CAN-XXXXXX",
      "generated_at": "ISO-TIMESTAMP",
      "role_target": "{role_target}",
      "deep_analysis": {{
        "ui_quality": {{ "score": 0-100, "evidence": "specific design elements", "comment": "1 sentence" }},
        "ux_thinking": {{ "score": 0-100, "evidence": "research process", "comment": "1 sentence" }},
        "project_maturity": {{ "score": 0-100, "evidence": "complexity", "comment": "1 sentence" }},
        "overall_score": 0-100,
        "summary": "2 sentences max"
      }},
      "skill_extractor": {{
        "skills": [
          {{ "name": "skill name", "category": "design_tool|dev_tool|methodology|soft_skill", "proficiency": "beginner|intermediate|advanced|expert", "evidence": "why" }}
        ],
        "primary_skills": ["top 3 skills"],
        "skill_gaps": ["expected but absent"],
        "total_skills_detected": 0
      }},
      "innovation_score": {{
        "originality_score": 0-100,
        "creative_risk_level": "low|medium|high",
        "standout_moments": [
          {{ "project": "name", "innovation": "unconventional thing", "impact": "why" }}
        ],
        "pattern_dependency": "reliance on templates",
        "innovation_summary": "2 sentences",
        "score_rationale": "1 sentence"
      }},
      "benchmarking": {{
        "industry_score": 0-100,
        "percentile_estimate": "top X%",
        "present": ["matching industry standard"],
        "partial": ["underdeveloped elements"],
        "missing": ["expected but absent"],
        "vs_top_portfolios": "1-2 sentences comparing to benchmarks",
        "industry_readiness": "ready_now|nearly_ready|needs_6_months|needs_1_year"
      }},
      "hidden_talent": {{
        "hidden_potential_score": 0-100,
        "presentation_vs_substance_gap": "small|medium|large",
        "hidden_signals": [
          {{ "signal": "found", "location": "where", "why_it_matters": "why" }}
        ],
        "overlooked_strengths": ["strengths"],
        "recruiter_flag": "must_interview|worth_interview|standard|pass",
        "flag_reason": "1 sentence"
      }},
      "portfolio_coach": {{
        "suggestions": [
          {{ "priority": "critical|high|medium|low", "area": "aspect", "problem": "what is wrong", "fix": "actionable fix", "example": "example" }}
        ],
        "quick_wins": ["3 changes taking <1 day"],
        "overall_coach_note": "2-3 sentences"
      }},
      "tech_depth": {{
        "technical_score": 0-100,
        "complexity_level": "basic|intermediate|advanced|expert",
        "tools_and_stack": ["tools"],
        "complexity_evidence": [
          {{ "project": "name", "technical_detail": "complexity", "sophistication": "impressive aspect" }}
                ],
        "technical_summary": "2 sentences",
        "technical_gaps": ["expected gap"]
      }},
      "consistency": {{
        "consistency_score": 0-100,
        "layout_coherence": {{ "score": 0-100, "issues": ["issues"] }},
        "brand_consistency": {{ "score": 0-100, "issues": ["issues"] }},
        "completeness": {{ "score": 0-100, "missing_sections": ["missing"] }},
        "professionalism_flag": "publication_ready|minor_polish_needed|needs_significant_work",
        "consistency_summary": "2 sentences"
      }},
      "weighted_final_score": 0-100,
      "score_breakdown": {{
        "deep_analysis": {{ "raw_score": 0-100, "weight": 0.15, "weighted_contribution": 0.0 }},
        "skill_extractor": {{ "raw_score": 0-100, "weight": 0.15, "weighted_contribution": 0.0 }},
        "innovation_score": {{ "raw_score": 0-100, "weight": 0.10, "weighted_contribution": 0.0 }},
        "benchmarking": {{ "raw_score": 0-100, "weight": 0.10, "weighted_contribution": 0.0 }},
        "hidden_talent": {{ "raw_score": 0-100, "weight": 0.10, "weighted_contribution": 0.0 }},
        "portfolio_coach": {{ "raw_score": 0-100, "weight": 0.15, "weighted_contribution": 0.0 }},
        "tech_depth": {{ "raw_score": 0-100, "weight": 0.15, "weighted_contribution": 0.0 }},
        "consistency": {{ "raw_score": 0-100, "weight": 0.10, "weighted_contribution": 0.0 }}
      }},
      "headline_scores": {{
        "portfolio_quality": 0-100,
        "creativity": "low|medium|high|exceptional",
        "problem_solving": "weak|developing|strong|exceptional",
        "industry_readiness": 0-100,
        "weighted_final_score": 0-100
      }},
      "detected_skills": {{
        "advanced": ["skills"],
        "intermediate": ["skills"],
        "beginner": ["skills"]
      }},
      "strengths": [
        {{ "point": "strength", "evidence": "portfolio reference" }}
      ],
      "weaknesses": [
        {{ "point": "weakness", "fix": "fix action" }}
      ],
      "hidden_talent_flag": "yes|no",
      "hidden_talent_note": "notes",
      "top_3_improvements": ["improvement1", "improvement2", "improvement3"],
      "recruiter_recommendation": "must_interview|recommend|consider|pass",
      "recommendation_reason": "2 sentences max",
      "coach_note": "3 sentences"
    }}
    """

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(prompt)
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error calling Gemini in analyzer: {e}. Trying OpenAI fallback.")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Portfolio Intelligence Agent API that evaluates portfolios and outputs valid JSON strictly matching templates."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            clean_text = response.choices[0].message.content.strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error calling OpenAI in analyzer: {e}. Falling back to heuristics.")

    return run_heuristic_analysis(text, filename, role_target, seniority)

