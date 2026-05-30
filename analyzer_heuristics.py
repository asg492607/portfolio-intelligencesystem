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


def run_heuristic_analysis(text: str, filename: str) -> dict:
    """Analyze text using heuristics to extract skills, design artifacts, and projects."""
    text_lower = text.lower()

    # ── Skill detection ─────────────────────────────────────────────────────────
    detected = {"design_tool": [], "dev_tool": [], "methodology": [], "soft_skill": []}
    for cat, list_of_words in TECH_KEYWORDS.items():
        for word in list_of_words:
            if word in text_lower or (word == "git" and re.search(r'\bgit\b', text_lower)):
                detected[cat].append(word.title() if len(word) > 3 else word.upper())

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
    if project_matches:
        for p_name in set(project_matches):
            p_name = p_name.strip()
            if p_name and len(p_name) > 3:
                extracted_projects.append({
                    "name": p_name.title(),
                    "type": "Case Study / Project",
                    "details": "Heuristically extracted project case study from text content.",
                    "images": []
                })
    else:
        # Fallback project placeholder
        extracted_projects.append({
            "name": "General Portfolio Project",
            "type": "Case Study",
            "details": "Extracted project from portfolio content.",
            "images": []
        })

    # ── Clean report structure ──────────────────────────────────────────────────
    report = {
        "report_id": str(uuid.uuid4()),
        "candidate_id": f"CAN-{str(uuid.uuid4())[:8].upper()}",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "skills": detected,
        "design_artifacts": {
            "artifacts_found": artifacts_found,
            "artifacts_missing": artifacts_missing
        },
        "projects": extracted_projects
    }
    return report
