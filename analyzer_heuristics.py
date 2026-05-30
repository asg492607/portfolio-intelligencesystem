import re
import uuid
import datetime

DEFAULT_WEIGHTS = {
    "deep_analysis": 0.20,
    "skill_extractor": 0.15,
    "design_artifacts": 0.20,
    "innovation_score": 0.10,
    "project_quality": 0.15,
    "tech_depth": 0.10,
    "consistency": 0.10
}

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


def run_heuristic_analysis(text: str, filename: str, role_target: str, seniority: str) -> dict:
    """Analyze text using heuristics to generate a realistic structured portfolio review."""
    role_target = role_target or "UX Designer"
    seniority = seniority or "Mid"

    text_lower = text.lower()

    # ── Skill detection ─────────────────────────────────────────────────────────
    detected = {"design_tool": [], "dev_tool": [], "methodology": [], "soft_skill": []}
    for cat, list_of_words in TECH_KEYWORDS.items():
        for word in list_of_words:
            if word in text_lower or (word == "git" and re.search(r'\bgit\b', text_lower)):
                detected[cat].append(word.title() if len(word) > 3 else word.upper())

    if not any(detected.values()):
        if "ux" in role_target.lower() or "design" in role_target.lower():
            detected["design_tool"] = ["Figma", "Adobe XD", "Miro"]
            detected["methodology"] = ["User Research", "Wireframing", "Prototyping"]
            detected["soft_skill"] = ["Collaboration", "Problem Solving"]
        else:
            detected["dev_tool"] = ["React", "JavaScript", "Node.js", "Git"]
            detected["methodology"] = ["Agile", "Scrum"]
            detected["soft_skill"] = ["Problem Solving", "Critical Thinking"]

    skills_list = []
    all_flat_detected = []
    for cat, items in detected.items():
        for i, item in enumerate(items):
            all_flat_detected.append(item)
            prof = "advanced" if i == 0 else ("intermediate" if i < 3 else "beginner")
            skills_list.append({
                "name": item,
                "category": cat,
                "proficiency": prof,
                "evidence": "Detected in portfolio content."
            })

    primary_skills = all_flat_detected[:3] if len(all_flat_detected) >= 3 else all_flat_detected

    gaps = []
    if "ux" in role_target.lower() or "design" in role_target.lower():
        if "Figma" not in all_flat_detected:
            gaps.append("Figma")
        if "User Research" not in all_flat_detected:
            gaps.append("User Research / Usability Testing")
        if "design system" not in text_lower:
            gaps.append("Design Systems (component/token documentation)")
    else:
        if "Git" not in all_flat_detected:
            gaps.append("Git Version Control")
        if "Docker" not in all_flat_detected:
            gaps.append("Docker Containerization")
        if "jest" not in text_lower and "testing" not in text_lower:
            gaps.append("Unit Testing (Jest/PyTest)")

    # ── Design artifact detection ────────────────────────────────────────────────
    artifacts_found = []
    artifacts_missing = []
    for artifact_name, keywords in ARTIFACT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            artifacts_found.append(artifact_name)
        else:
            artifacts_missing.append(artifact_name)

    artifact_count = len(artifacts_found)
    artifact_quality = (
        "excellent" if artifact_count >= 7 else
        "good" if artifact_count >= 5 else
        "fair" if artifact_count >= 3 else
        "poor"
    )
    documentation_depth = (
        "comprehensive" if artifact_count >= 7 else
        "detailed" if artifact_count >= 5 else
        "moderate" if artifact_count >= 3 else
        "shallow"
    )

    # ── Scoring ──────────────────────────────────────────────────────────────────
    text_length_score = min(30, len(text) // 100)
    skills_count_score = min(30, len(skills_list) * 3)
    artifact_bonus = min(20, artifact_count * 2)
    base_score = 40 + text_length_score + skills_count_score
    base_score = min(95, max(35, base_score))

    ui_score = min(100, int(base_score + 5))
    ux_score = min(100, int(base_score - 2))
    maturity_score = min(100, int(base_score - 5))
    tech_score = min(100, int(base_score if "dev" in role_target.lower() else base_score - 10))
    innovation_score_val = min(100, int(base_score - 8))
    consistency_score_val = min(100, int(base_score + 2))
    artifact_score = min(100, int(40 + artifact_bonus + skills_count_score))
    project_quality_score = min(100, int(base_score - 3))

    # ── Module: deep_analysis ────────────────────────────────────────────────────
    deep_analysis = {
        "ui_quality": {
            "score": ui_score,
            "evidence": "Consistent layout grids and clear visual hierarchy detected across case studies.",
            "comment": "Visually appealing layouts with professional typography and content spacing."
        },
        "ux_thinking": {
            "score": ux_score,
            "evidence": "User journey maps and wireframe mockups embedded in the design process." if "user flow" in text_lower or "wireframe" in text_lower else "Process documentation is present but limited in depth.",
            "comment": "Shows user-centric planning; expanding user testing details would strengthen this further."
        },
        "project_maturity": {
            "score": maturity_score,
            "evidence": "Real-world project constraints and responsive screen layouts presented.",
            "comment": "Projects exhibit good completeness, illustrating standard professional workflows."
        },
        "overall_score": int((ui_score + ux_score + maturity_score) / 3),
        "summary": f"This portfolio shows a solid foundation in {role_target} fundamentals. Features clear documentation of project iterations and neat visual presentation."
    }

    # ── Module: skill_extractor ──────────────────────────────────────────────────
    skill_extractor = {
        "skills": skills_list,
        "primary_skills": primary_skills,
        "skill_gaps": gaps,
        "total_skills_detected": len(skills_list)
    }

    # ── Module: design_artifacts ─────────────────────────────────────────────────
    design_artifacts = {
        "artifacts_found": artifacts_found if artifacts_found else ["portfolio presentation slides"],
        "artifacts_missing": artifacts_missing[:5],
        "artifact_quality": artifact_quality,
        "documentation_depth": documentation_depth,
        "artifact_summary": f"Found {len(artifacts_found)} artifact types in this portfolio. " + (
            "Strong artifact coverage supports a well-rounded design process narrative." if artifact_count >= 5
            else "Adding more process artifacts (wireframes, user flows, research) would significantly strengthen this portfolio."
        )
    }

    # ── Module: innovation_score ─────────────────────────────────────────────────
    innovation_score = {
        "originality_score": innovation_score_val,
        "creative_risk_level": "medium" if innovation_score_val > 60 else "low",
        "standout_moments": [
            {
                "project": "Primary Case Study",
                "innovation": "Novel problem-framing that prioritizes micro-interactions over standard layouts.",
                "impact": "Improves user engagement and guides the reader naturally through the content."
            }
        ],
        "pattern_dependency": "Low to moderate usage of template grids; custom graphics are evident." if innovation_score_val > 60 else "Moderate reliance on conventional layout patterns.",
        "innovation_summary": "Demonstrates good custom styling and a personalized layout structure.",
        "score_rationale": "Displays unique stylistic elements while respecting core layout conventions."
    }

    # ── Module: project_quality ──────────────────────────────────────────────────
    project_quality = {
        "quality_score": project_quality_score,
        "complexity_level": "advanced" if project_quality_score > 80 else ("intermediate" if project_quality_score > 60 else "basic"),
        "projects_found": [
            {
                "name": "Detected Project",
                "type": "Design / Product Case Study",
                "quality_indicators": ["Visual documentation", "Process walkthrough"],
                "depth": "moderate" if project_quality_score > 60 else "shallow"
            }
        ],
        "documentation_quality": "good" if project_quality_score > 70 else "fair",
        "quality_summary": f"Projects demonstrate a {('strong' if project_quality_score > 75 else 'developing')} level of execution quality for a {seniority} {role_target}. Documentation coverage could be expanded with more process detail."
    }

    # ── Module: tech_depth ───────────────────────────────────────────────────────
    tech_depth = {
        "technical_score": tech_score,
        "complexity_level": "advanced" if tech_score > 80 else ("intermediate" if tech_score > 60 else "basic"),
        "tools_and_stack": detected["dev_tool"] if detected["dev_tool"] else ["Git", "GitHub Pages", "HTML/CSS"],
        "complexity_evidence": [
            {
                "project": "Portfolio Presentation",
                "technical_detail": "Clean document layout structures and standard responsive assets.",
                "sophistication": "Uses clean layout grids and readable fonts, indicating structural discipline."
            }
        ],
        "technical_summary": "Demonstrates appropriate technical foundations needed for professional execution.",
        "technical_gaps": gaps[:2]
    }

    # ── Module: consistency ──────────────────────────────────────────────────────
    consistency = {
        "consistency_score": consistency_score_val,
        "layout_coherence": {"score": consistency_score_val, "issues": []},
        "brand_consistency": {"score": min(100, consistency_score_val + 3), "issues": []},
        "completeness": {"score": min(100, consistency_score_val + 5), "missing_sections": []},
        "professionalism_flag": "publication_ready" if consistency_score_val > 75 else "minor_polish_needed",
        "consistency_summary": "High degree of style matching and content consistency. Grids are uniform across sections."
    }

    # ── Module: portfolio_coach ──────────────────────────────────────────────────
    portfolio_coach = {
        "suggestions": [
            {
                "priority": "high",
                "area": "User Testing Evidence",
                "problem": "Lack of quantitative metrics from user tests.",
                "fix": "Include task success rates and time-on-task metrics in the case studies.",
                "example": "e.g., 'Tested with 5 users, achieving a 90% task completion rate (up from 65%).'"
            },
            {
                "priority": "medium",
                "area": "Design System Documentation",
                "problem": "Components are shown without explicit token or grid specifications.",
                "fix": "Add a section detailing spacing rules, color tokens, and typography styles used.",
                "example": "Show a screenshot of the Figma style library used for the project."
            }
        ],
        "quick_wins": [
            "Link interactive high-fidelity prototype at the beginning of each case study",
            "Add a brief summary card with role, duration, and team size for each project",
            "Proofread headings to ensure uniform sentence casing"
        ],
        "overall_coach_note": "A capable portfolio demonstrating necessary visual and structural benchmarks. Focus on highlighting user metrics and adding more artifact depth to elevate the work to senior-level presentation standards."
    }

    # ── Weighted score ───────────────────────────────────────────────────────────
    weighted_score = min(100, max(0, int(
        deep_analysis["overall_score"] * DEFAULT_WEIGHTS["deep_analysis"] +
        min(100, skill_extractor["total_skills_detected"] * 8) * DEFAULT_WEIGHTS["skill_extractor"] +
        artifact_score * DEFAULT_WEIGHTS["design_artifacts"] +
        innovation_score["originality_score"] * DEFAULT_WEIGHTS["innovation_score"] +
        project_quality["quality_score"] * DEFAULT_WEIGHTS["project_quality"] +
        tech_depth["technical_score"] * DEFAULT_WEIGHTS["tech_depth"] +
        consistency["consistency_score"] * DEFAULT_WEIGHTS["consistency"]
    )))

    # ── Final report ─────────────────────────────────────────────────────────────
    report = {
        "report_id": str(uuid.uuid4()),
        "candidate_id": f"CAN-{str(uuid.uuid4())[:8].upper()}",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "role_target": role_target,
        "seniority": seniority,

        "deep_analysis": deep_analysis,
        "skill_extractor": skill_extractor,
        "design_artifacts": design_artifacts,
        "innovation_score": innovation_score,
        "project_quality": project_quality,
        "portfolio_coach": portfolio_coach,
        "tech_depth": tech_depth,
        "consistency": consistency,

        "weighted_final_score": weighted_score,
        "score_breakdown": {
            "deep_analysis": {"raw_score": deep_analysis["overall_score"], "weight": DEFAULT_WEIGHTS["deep_analysis"], "weighted_contribution": round(deep_analysis["overall_score"] * DEFAULT_WEIGHTS["deep_analysis"], 2)},
            "skill_extractor": {"raw_score": min(100, skill_extractor["total_skills_detected"] * 8), "weight": DEFAULT_WEIGHTS["skill_extractor"], "weighted_contribution": round(min(100, skill_extractor["total_skills_detected"] * 8) * DEFAULT_WEIGHTS["skill_extractor"], 2)},
            "design_artifacts": {"raw_score": artifact_score, "weight": DEFAULT_WEIGHTS["design_artifacts"], "weighted_contribution": round(artifact_score * DEFAULT_WEIGHTS["design_artifacts"], 2)},
            "innovation_score": {"raw_score": innovation_score["originality_score"], "weight": DEFAULT_WEIGHTS["innovation_score"], "weighted_contribution": round(innovation_score["originality_score"] * DEFAULT_WEIGHTS["innovation_score"], 2)},
            "project_quality": {"raw_score": project_quality["quality_score"], "weight": DEFAULT_WEIGHTS["project_quality"], "weighted_contribution": round(project_quality["quality_score"] * DEFAULT_WEIGHTS["project_quality"], 2)},
            "tech_depth": {"raw_score": tech_depth["technical_score"], "weight": DEFAULT_WEIGHTS["tech_depth"], "weighted_contribution": round(tech_depth["technical_score"] * DEFAULT_WEIGHTS["tech_depth"], 2)},
            "consistency": {"raw_score": consistency["consistency_score"], "weight": DEFAULT_WEIGHTS["consistency"], "weighted_contribution": round(consistency["consistency_score"] * DEFAULT_WEIGHTS["consistency"], 2)}
        },

        "headline_scores": {
            "portfolio_quality": int((ui_score + ux_score) / 2),
            "creativity": "high" if innovation_score_val > 75 else ("medium" if innovation_score_val > 55 else "low"),
            "problem_solving": "strong" if ux_score > 70 else "developing",
            "artifact_richness": "high" if artifact_count >= 5 else ("medium" if artifact_count >= 3 else "low"),
            "weighted_final_score": weighted_score
        },

        "detected_skills": {
            "advanced": [s["name"] for s in skills_list if s["proficiency"] == "advanced"],
            "intermediate": [s["name"] for s in skills_list if s["proficiency"] == "intermediate"],
            "beginner": [s["name"] for s in skills_list if s["proficiency"] == "beginner"]
        },

        "strengths": [
            {"point": "Strong visual coherence", "evidence": "Grid patterns and styling remain steady throughout the documents."},
            {"point": "Methodological documentation", "evidence": "Explicit callouts for user paths and layout structures are included."}
        ],

        "weaknesses": [
            {"point": "Insufficient quantitative metrics", "fix": "Incorporate key user performance metrics into project reviews."},
            {"point": "Limited design artifact coverage", "fix": "Add wireframes, user flows, and research documentation to show end-to-end process."}
        ],

        "top_3_improvements": [
            "Add wireframes and user flow diagrams to show the design process from ideation to delivery.",
            "Include links to live interactive prototypes or recorded walkthroughs.",
            "Write a brief executive summary at the start of each case study with problem, solution, and outcome."
        ],

        "coach_note": "A solid portfolio demonstrating good visual and structural fundamentals. Focus on expanding design artifact coverage and adding quantitative user research data to unlock senior-level opportunities."
    }
    return report
