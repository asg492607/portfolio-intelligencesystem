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

# Default weights — 7-module real engine
DEFAULT_WEIGHTS = {
    "deep_analysis": 0.20,
    "skill_extractor": 0.15,
    "design_artifacts": 0.20,
    "innovation_score": 0.10,
    "project_quality": 0.15,
    "tech_depth": 0.10,
    "consistency": 0.10
}


# 1. PDF parsing using PyMuPDF (fitz)
def extract_text_from_pdf(file_path: str, job_id: str = None) -> str:
    text_content = ""
    try:
        if use_fitz:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # If job_id is provided, extract images on the fly and embed placeholders
                page_images_placeholders = ""
                if job_id:
                    os.makedirs(f"local_storage/{job_id}", exist_ok=True)
                    image_list = page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        img_filename = f"{job_id}/extracted_img_{page_num + 1}_{img_idx + 1}.{image_ext}"
                        
                        from storage import storage_client
                        url = storage_client.upload_data(image_bytes, img_filename, f"image/{image_ext}")
                        
                        if url.startswith("local_storage/"):
                            url = "/" + url
                        
                        page_images_placeholders += f"\n[IMAGE_URL: {url} CAPTION: Page {page_num + 1} Image {img_idx + 1}]\n"
                
                text_content += page_text + "\n" + page_images_placeholders + "\n"
        else:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text_content.strip()

def extract_images_from_pdf(file_path: str, job_id: str) -> list:
    # Deprecated/Handled inline by extract_text_from_pdf to inject placeholders.
    # Return empty list to prevent duplicate logic execution
    return []

# 2. Web Scraping for Website / Behance (Enhanced Accuracy Boilerplate Stripping)
async def scrape_url_content(url: str) -> tuple:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract image URLs and insert inline text markers
                images = []
                for img_tag in soup.find_all("img"):
                    src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-hi-res") or img_tag.get("srcset")
                    alt = img_tag.get("alt") or ""
                    if src:
                        if "," in src:
                            src = src.split(",")[0].strip().split(" ")[0]
                        from urllib.parse import urljoin
                        absolute_url = urljoin(url, src)
                        if absolute_url.startswith("http") and not any(x in absolute_url.lower() for x in ["pixel", "analytics", "tracker", "sprite", "logo", "icon", "svg"]):
                            images.append(absolute_url)
                            
                            # Inject placeholder
                            placeholder = soup.new_tag("p")
                            placeholder.string = f"\n[IMAGE_URL: {absolute_url} CAPTION: {alt}]\n"
                            img_tag.insert_after(placeholder)
                            
                            if len(images) >= 15:
                                break
                
                # Extract links
                links = []
                for a_tag in soup.find_all("a"):
                    href = a_tag.get("href")
                    if href:
                        from urllib.parse import urljoin
                        absolute_url = urljoin(url, href)
                        if absolute_url.startswith("http") and not any(x in absolute_url.lower() for x in ["facebook", "twitter", "linkedin", "instagram", "youtube", "pinterest", "reddit"]):
                            links.append(absolute_url)
                            if len(links) >= 15:
                                break

                # Extract meta description for high-level context
                meta_desc = ""
                meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
                if meta_tag:
                    meta_desc = meta_tag.get("content", "").strip()

                # Strip structural navigation, scripts, styles, headers, and footers to isolate project body text
                for noise in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                    noise.extract()
                
                # Further purge generic elements by class/id matching typical template noise
                for class_noise in soup.find_all(class_=re.compile(r"footer|header|menu|nav|sidebar|copyright|cookie|social|advert", re.IGNORECASE)):
                    class_noise.extract()
                
                # Extract text contents
                lines = (line.strip() for line in soup.get_text().splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                title = soup.title.string if soup.title else "Scraped Portfolio"
                
                context_str = f"URL: {url}\nTitle: {title}\n"
                if meta_desc:
                    context_str += f"Meta Description: {meta_desc}\n"
                return f"{context_str}Content:\n{text[:18000]}", images, links
            else:
                return f"Failed to retrieve URL {url}. Status code: {response.status_code}", [], []
    except Exception as e:
        return f"Error occurred scraping URL {url}: {str(e)}", [], []

# 3. Figma API Parser (Enhanced to extract Structural Design Artifact Signals)
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
                detected_artifacts = set()
                styles_count = len(data.get("styles", {}))
                components_count = len(data.get("components", {}))
                
                # Mapping of figma layer names to design artifacts (to help AI evaluate with 100% accuracy)
                artifact_indicators = {
                    "wireframe": "Wireframe / Lo-fi Screen",
                    "user flow": "User Flow / Journey Map",
                    "prototype": "Interactive Prototype",
                    "persona": "User Persona Profile",
                    "moodboard": "Inspiration Moodboard",
                    "style guide": "Style Guide / Token Sheet",
                    "design system": "Design System Library",
                    "mockup": "Hi-fi Mockup Screen",
                    "usability": "Usability Testing Results"
                }

                # Helper to traverse Figma JSON Document Nodes
                def traverse_nodes(node):
                    node_name = str(node.get("name", "")).lower()
                    
                    # Inspect layer name to check if it matches design artifacts
                    for keyword, tag in artifact_indicators.items():
                        if keyword in node_name:
                            detected_artifacts.add(tag)

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
                artifacts_str = ", ".join(detected_artifacts) if detected_artifacts else "None directly labeled in frame hierarchy"
                
                return f"Figma File: {data.get('name', 'Unnamed')}\nComponents count: {components_count}\nStyles count: {styles_count}\nStructural Artifact Signals: [{artifacts_str}]\nContent:\n{compiled_text}"
            else:
                return f"Figma API error. Status: {response.status_code}. Detail: {response.text}"
    except Exception as e:
        return f"Error calling Figma API: {str(e)}"

# 4. Generate Embeddings (Optional Vector representation)
async def generate_text_embedding(text: str) -> list:
    """Generate embedding vector (returns dummy vector to remove external API dependency)."""
    # Dummy embedding fallback matching Qdrant size
    import random
    return [random.uniform(-0.1, 0.1) for _ in range(768)]

# Heuristics local engine fallback (already built, import same)
from analyzer_heuristics import run_heuristic_analysis

def run_ai_analysis(text: str, filename: str, images: list = None, links: list = None) -> dict:
    """Runs data extraction using local Ollama LLM (llama3.1), or falls back to heuristics."""
    from openai import OpenAI

    prompt = f"""
    You are Portfolio Ingestion Agent — an AI system that analyzes portfolios to extract candidate profiles, technology stack tools, identify design artifacts, and list projects.
    Your task is to analyze the portfolio content below and extract structured data. Focus strictly on objective data extraction; do not include ratings, reviews, recommendations, or grading of any kind.
    
    IMPORTANT: Look for inline markers like `[IMAGE_URL: <url> CAPTION: <text>]` inside the text stream. Identify which images belong to which projects, and assign those exact image URLs to the corresponding project in the "projects" array below.

    Source Context: {filename}

    Portfolio text content:
    ---
    {text[:15000]}
    ---

    Return a JSON object matching EXACTLY this structure. Output only valid JSON — no markdown, no preambles:
    {{
      "report_id": "unique-uuid",
      "candidate_id": "CAN-XXXXXX",
      "generated_at": "ISO-TIMESTAMP",
      "full_name": "candidate full name or empty string if not found",
      "headline": "candidate professional headline or role title",
      "summary": "a brief professional summary/bio summarizing their background",
      "target_roles": ["list of target roles/positions they want or have"],
      "years_experience": float or null for years of experience,
      "industries": ["list of industries they worked in or design for"],
      "strengths": ["list of candidate's core strengths/qualities"],
      "tools": ["list of tools/technologies mentioned in the portfolio"],
      "skills": {{
        "design_tool": ["list of design tools found, e.g. Figma, Photoshop, Sketch"],
        "dev_tool": ["list of development tools/languages, e.g. React, Python, Docker"],
        "methodology": ["list of methodologies, e.g. Wireframing, User Research, Agile"],
        "soft_skill": ["list of soft skills, e.g. Collaboration, Problem Solving"]
      }},
      "design_artifacts": {{
        "artifacts_found": ["list of identified design artifacts e.g. wireframes, mockups, case studies, user flows, prototypes, design systems, style guides"],
        "artifacts_missing": ["expected design artifacts not found in the content"]
      }},
      "projects": [
        {{ 
          "name": "project name", 
          "type": "type of project, e.g. Mobile App, E-Commerce Website, Branding", 
          "details": "brief description of project scope, technologies used, and contribution details",
          "images": ["list of matching IMAGE_URL strings found in the text for this project"]
        }}
      ]
    }}
    """

    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if groq_key:
        try:
            # Use Groq Cloud Llama API
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key
            )
            response = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": "You are a Portfolio Ingestion Agent API that extracts structured candidate profile data from portfolios and outputs valid JSON matching templates."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            clean_text = response.choices[0].message.content.strip()
            match = re.search(r'\{[\s\S]*\}', clean_text)
            if match:
                clean_text = match.group(0)
            
            result = json.loads(clean_text)
            
            # Guarantee fallback keys
            if "candidate_id" not in result or not result["candidate_id"]:
                result["candidate_id"] = f"CAN-{str(uuid.uuid4())[:8].upper()}"
            if "report_id" not in result or not result["report_id"]:
                result["report_id"] = str(uuid.uuid4())
            if "generated_at" not in result or not result["generated_at"]:
                result["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                
            return result
        except Exception as e:
            print(f"Error calling Groq model in analyzer: {e}. Trying heuristics fallback.")

    return run_heuristic_analysis(text, filename, images=images)


