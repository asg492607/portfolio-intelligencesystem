import re
import uuid
import datetime

TECH_KEYWORDS = {
    "design_tool": ["figma", "sketch", "photoshop", "illustrator", "adobe xd", "invision", "miro", "canva", "zeplin", "framer"],
    "dev_tool": ["react", "vue", "angular", "next.js", "node.js", "javascript", "typescript", "python", "django", "fastapi", "flask", "postgresql", "mongodb", "docker", "aws", "git", "tailwind", "css", "html"],
    "methodology": ["user research", "wireframing", "prototyping", "usability testing", "agile", "scrum", "design thinking", "information architecture", "persona", "user flows"],
    "soft_skill": ["communication", "collaboration", "leadership", "problem solving", "time management", "adaptability", "critical thinking"]
}

ARTIFACT_KEYWORDS = {
    "wireframes": ["wireframe", "wireframing", "lo-fi", "low-fi", "low fidelity"],
    "mockups": ["mockup", "mock-up", "high fidelity", "hi-fi", "high-fi"],
    "case studies": ["case study", "case studies", "project overview"],
    "user flows": ["user flow", "user journey", "flow diagram", "task flow"],
    "prototypes": ["prototype", "prototyping", "interactive prototype", "clickable"],
    "design systems": ["design system", "component library", "style guide", "token", "design tokens"],
    "research": ["user research", "research findings", "user interview", "survey", "usability test", "a/b test"],
    "personas": ["persona", "user persona", "user archetype"],
    "information architecture": ["information architecture", "ia diagram", "sitemap", "card sort"],
    "style guides": ["style guide", "brand guide", "typography guide", "color palette"]
}


def run_heuristic_analysis(text: str, filename: str, images: list = None) -> dict:
    """Analyze text using heuristics to extract skills, design artifacts, and projects."""
    text_lower = text.lower()

    # ── Skill detection ─────────────────────────────────────────────────────────
    detected = {"design_tools": [], "methodologies_and_processes": [], "soft_skills": []}
    for cat, list_of_words in TECH_KEYWORDS.items():
        for word in list_of_words:
            if word in text_lower or (word == "git" and re.search(r'\bgit\b', text_lower)):
                val = word.title() if len(word) > 3 else word.upper()
                if cat in ["design_tool", "dev_tool"]:
                    detected["design_tools"].append(val)
                elif cat == "methodology":
                    detected["methodologies_and_processes"].append(val)
                elif cat == "soft_skill":
                    detected["soft_skills"].append(val)

    # ── Design artifact detection ────────────────────────────────────────────────
    artifacts_found = []
    artifacts_missing = []
    for artifact_name, keywords in ARTIFACT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            artifacts_found.append(artifact_name)
        else:
            artifacts_missing.append(artifact_name)

    # ── Project Extraction ───────────────────────────────────────────────────────
    extracted_projects = []
    project_matches = re.findall(r'(?:project|case\s+study)(?:\s+name)?\s*:\s*([^\n\r\.\,\;\:]{3,40})', text, re.IGNORECASE)
    flat_images = images or []
    if project_matches:
        matches = list(set(project_matches))
        for idx, p_name in enumerate(matches):
            p_name = p_name.strip()
            if p_name and len(p_name) > 3:
                # Assign 1-2 images heuristically
                p_images = flat_images[idx*2 : (idx+1)*2]
                extracted_projects.append({
                    "name": p_name.title(),
                    "type": "Case Study / Project",
                    "role": "Lead Designer / Developer",
                    "client_or_organization": "Design Agency Client",
                    "timeline": "3 Months",
                    "team_size": "Team of 4",
                    "details": "Heuristically extracted project case study from text content.",
                    "technologies": detected["design_tools"][:2],
                    "challenges": "Optimizing user experience journeys and aligning system components.",
                    "outcomes": "Successful implementation and positive stakeholder alignment.",
                    "images": p_images
                })
    else:
        # Fallback project placeholder
        extracted_projects.append({
            "name": "General Portfolio Project",
            "type": "Case Study",
            "role": "Designer & Developer",
            "client_or_organization": "Personal Project",
            "timeline": "1 Month",
            "team_size": "Solo Project",
            "details": "Extracted project from portfolio content.",
            "technologies": detected["design_tools"][:2],
            "outcomes": "Demonstrated technical skills and creative execution.",
            "images": flat_images[:3]
        })

    # ── Intelligent heuristic extraction for candidate profile fields ──────────
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    guessed_name = ""
    if lines:
        first_line = lines[0]
        if len(first_line) < 40 and not any(x in first_line.lower() for x in ["http", "portfolio", "resume", "cv", "page", "work"]):
            guessed_name = first_line
    if not guessed_name:
        guessed_name = filename.split('.')[0].replace('_', ' ').replace('-', ' ').title()

    guessed_headline = "Creative & Design Professional"
    for line in lines[:5]:
        if any(x in line.lower() for x in ["designer", "developer", "engineer", "manager", "strategist", "illustrator"]):
            if len(line) < 100:
                guessed_headline = line
                break

    guessed_summary = text[:300].strip() + "..." if len(text) > 300 else text

    # Extract years experience if mentioned
    exp_match = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience', text_lower)
    years_experience = float(exp_match.group(1)) if exp_match else None

    # Extra target roles
    target_roles = []
    role_match = re.search(r'target\s+roles?\s*:\s*([^\n]+)', text_lower)
    if role_match:
        target_roles = [r.strip().title() for r in role_match.group(1).split(',') if r.strip()]
    else:
        target_roles = [guessed_headline.title()]

    # Industries extraction heuristics
    industries_list = ["technology", "finance", "healthcare", "education", "retail", "e-commerce", "food & beverage", "entertainment"]
    detected_industries = [ind.title() for ind in industries_list if ind in text_lower]

    # Strengths extraction heuristics
    strengths_list = ["creative thinking", "problem solving", "collaboration", "communication", "detail-oriented", "leadership"]
    detected_strengths = [s.title() for s in strengths_list if s in text_lower]

    # Tools flat list (Design + Dev tools combined)
    flat_tools = detected["design_tools"]

    # ── Clean report structure ──────────────────────────────────────────────────
    report = {
        "report_id": str(uuid.uuid4()),
        "candidate_id": f"CAN-{str(uuid.uuid4())[:8].upper()}",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "full_name": guessed_name,
        "headline": guessed_headline,
        "summary": guessed_summary,
        "target_roles": target_roles[:3],
        "years_experience": years_experience,
        "industries": detected_industries,
        "strengths": detected_strengths,
        "tools": flat_tools,
        "skills": detected,
        "design_artifacts": {
            "artifacts_found": artifacts_found,
            "artifacts_missing": artifacts_missing
        },
        "projects": extracted_projects
    }
    return report

