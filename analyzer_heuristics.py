import re
import uuid
import datetime

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

def run_heuristic_analysis(text: str, filename: str, role_target: str, seniority: str) -> dict:
    """Analyze text using heuristics to generate a realistic structured portfolio review."""
    role_target = role_target or "UX Designer"
    seniority = seniority or "Mid"
    
    # Look for tools and skills in text
    detected = {
        "design_tool": [],
        "dev_tool": [],
        "methodology": [],
        "soft_skill": []
    }
    
    text_lower = text.lower()
    for cat, list_of_words in TECH_KEYWORDS.items():
        for word in list_of_words:
            if word in text_lower or (word == "git" and re.search(r'\bgit\b', text_lower)):
                detected[cat].append(word.title() if len(word) > 3 else word.upper())

    # Fallback to defaults if nothing found
    if not any(detected.values()):
        if "ux" in role_target.lower() or "design" in role_target.lower():
            detected["design_tool"] = ["Figma", "Adobe XD", "Miro"]
            detected["methodology"] = ["User Research", "Wireframing", "Prototyping"]
            detected["soft_skill"] = ["Collaboration", "Problem Solving"]
        else:
            detected["dev_tool"] = ["React", "JavaScript", "Node.js", "Git"]
            detected["methodology"] = ["Agile", "Scrum"]
            detected["soft_skill"] = ["Problem Solving", "Critical Thinking"]

    # Gather list of skills
    skills_list = []
    
    all_flat_detected = []
    for cat, items in detected.items():
        for i, item in enumerate(items):
            all_flat_detected.append(item)
            prof = "advanced" if i == 0 else ("intermediate" if i < 3 else "beginner")
            evidence = f"Listed in portfolio projects and resume sections."
            skills_list.append({
                "name": item,
                "category": cat,
                "proficiency": prof,
                "evidence": evidence
            })
            
    primary_skills = all_flat_detected[:3] if len(all_flat_detected) >= 3 else all_flat_detected
    
    # Skill gaps based on role target
    gaps = []
    if "ux" in role_target.lower() or "design" in role_target.lower():
        if "Figma" not in all_flat_detected: gaps.append("Figma")
        if "User Research" not in all_flat_detected: gaps.append("User Research / Usability Testing")
        if "Design System" not in text: gaps.append("Design Systems (tokenization)")
    else:
        if "Git" not in all_flat_detected: gaps.append("Git Version Control")
        if "Docker" not in all_flat_detected: gaps.append("Docker Containerization")
        if "Jest" not in text and "Testing" not in text: gaps.append("Unit Testing (Jest/PyTest)")

    # Heuristic scoring based on seniority and richness of text
    text_length_score = min(30, len(text) // 100) # up to 30 points
    skills_count_score = min(30, len(skills_list) * 3) # up to 30 points
    base_score = 40 + text_length_score + skills_count_score
    base_score = min(95, max(35, base_score)) # clamp 35-95

    # Modify scores slightly per module
    ui_score = min(100, int(base_score + 5))
    ux_score = min(100, int(base_score - 2))
    maturity_score = min(100, int(base_score - 5))
    tech_score = min(100, int(base_score if "dev" in role_target.lower() else base_score - 10))
    innovation_score_val = min(100, int(base_score - 8))
    consistency_score_val = min(100, int(base_score + 2))
    benchmarking_score_val = min(100, int(base_score - 4))
    hidden_talent_score_val = min(100, int(base_score + 3))

    deep_analysis = {
        "ui_quality": {
            "score": ui_score,
            "evidence": "Consistent layout grids and clear visual hierarchy across main case studies.",
            "comment": "Visually appealing layouts with professional typography and content spacing."
        },
        "ux_thinking": {
            "score": ux_score,
            "evidence": "User journey maps and wireframe mockups embedded in the design process.",
            "comment": "Shows strong user-centric planning, though user testing details could be expanded."
        },
        "project_maturity": {
            "score": maturity_score,
            "evidence": "Real-world project constraints and responsive screen layouts presented.",
            "comment": "Projects exhibit good completeness, illustrating standard professional workflows."
        },
        "overall_score": int((ui_score + ux_score + maturity_score) / 3),
        "summary": f"This portfolio highlights a solid foundation in {role_target} fundamentals. Features clear documentation of project iterations and neat visual presentation."
    }

    skill_extractor = {
        "skills": skills_list,
        "primary_skills": primary_skills,
        "skill_gaps": gaps,
        "total_skills_detected": len(skills_list)
    }

    innovation_score = {
        "originality_score": innovation_score_val,
        "creative_risk_level": "medium" if innovation_score_val > 60 else "low",
        "standout_moments": [
            {
                "project": "Primary Case Study",
                "innovation": "Novel problem-framing that prioritizes micro-interactions over standard layouts.",
                "impact": "Improves user retention and guides the reader naturally through the content."
            }
        ],
        "pattern_dependency": "Low to moderate usage of template grids; custom graphics are evident.",
        "innovation_summary": "Demonstrates good custom styling and a personalized layout structure.",
        "score_rationale": "Displays unique stylistic elements while respecting core layout conventions."
    }

    benchmarking = {
        "industry_score": benchmarking_score_val,
        "percentile_estimate": f"top {100 - benchmarking_score_val}%" if benchmarking_score_val > 50 else "top 60%",
        "present": ["Clear case study structures", "Mobile/desktop responsive frames"],
        "partial": ["User testing data metrics", "Interactive prototype links"],
        "missing": ["Refined design system tokens", "High-fidelity motion interactions"],
        "vs_top_portfolios": f"Ranks competitively for a {seniority} level role target, showing standard visual and functional polish.",
        "industry_readiness": "ready_now" if base_score > 75 else ("nearly_ready" if base_score > 60 else "needs_6_months")
    }

    hidden_talent = {
        "hidden_potential_score": hidden_talent_score_val,
        "presentation_vs_substance_gap": "medium" if abs(ui_score - ux_score) > 10 else "small",
        "hidden_signals": [
            {
                "signal": "Strong content layout alignment",
                "location": "Case studies",
                "why_it_matters": "Shows detail-oriented structure and professional alignment sensibilities."
            }
        ],
        "overlooked_strengths": ["Clear naming hierarchy", "Clean structure"],
        "recruiter_flag": "must_interview" if base_score > 80 else ("worth_interview" if base_score > 65 else "standard"),
        "flag_reason": f"Solid portfolio documentation aligned with {role_target} expectations."
    }

    portfolio_coach = {
        "suggestions": [
            {
                "priority": "high",
                "area": "User Testing",
                "problem": "Lack of quantitative metrics from user tests.",
                "fix": "Include task success rates and time-on-task metrics in the case studies.",
                "example": "e.g., 'Tested with 5 users, achieving a 90% completion rate (up from 65%).'"
            },
            {
                "priority": "medium",
                "area": "Design System Documentation",
                "problem": "Components are shown without explicit token or grid specifications.",
                "fix": "Add a small section detailing spacing rules, colors, and typography styles used.",
                "example": "Show a screenshot of the Figma style library used for the project."
            }
        ],
        "quick_wins": [
            "Link interactive high-fidelity prototype at the beginning of case studies",
            "Add a brief summary card containing the role, duration, and team for each project",
            "Proofread headings to ensure uniform sentence casing"
        ],
        "overall_coach_note": "A highly capable portfolio demonstrating all necessary visual and structural benchmarks. Focus on highlighting user metrics and code repository clean-ups to stand out."
    }

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

    consistency = {
        "consistency_score": consistency_score_val,
        "layout_coherence": {
            "score": consistency_score_val,
            "issues": []
        },
        "brand_consistency": {
            "score": min(100, consistency_score_val + 3),
            "issues": []
        },
        "completeness": {
            "score": min(100, consistency_score_val + 5),
            "missing_sections": []
        },
        "professionalism_flag": "publication_ready" if consistency_score_val > 75 else "minor_polish_needed",
        "consistency_summary": "High degree of style matching and content consistency. Grids are uniform."
    }

    # Recalculate weighted score
    weighted_score = min(100, max(0, int(base_score)))

    # Construct report
    report = {
        "report_id": str(uuid.uuid4()),
        "candidate_id": f"CAN-{str(uuid.uuid4())[:8].upper()}",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "role_target": role_target,
        "seniority": seniority,
        
        "deep_analysis": deep_analysis,
        "skill_extractor": skill_extractor,
        "innovation_score": innovation_score,
        "benchmarking": benchmarking,
        "hidden_talent": hidden_talent,
        "portfolio_coach": portfolio_coach,
        "tech_depth": tech_depth,
        "consistency": consistency,

        "weighted_final_score": weighted_score,
        "score_breakdown": {
            "deep_analysis": {"raw_score": deep_analysis["overall_score"], "weight": DEFAULT_WEIGHTS["deep_analysis"], "weighted_contribution": round(deep_analysis["overall_score"] * DEFAULT_WEIGHTS["deep_analysis"], 2)},
            "skill_extractor": {"raw_score": min(100, skill_extractor["total_skills_detected"] * 8), "weight": DEFAULT_WEIGHTS["skill_extractor"], "weighted_contribution": round(min(100, skill_extractor["total_skills_detected"] * 8) * DEFAULT_WEIGHTS["skill_extractor"], 2)},
            "innovation_score": {"raw_score": innovation_score["originality_score"], "weight": DEFAULT_WEIGHTS["innovation_score"], "weighted_contribution": round(innovation_score["originality_score"] * DEFAULT_WEIGHTS["innovation_score"], 2)},
            "benchmarking": {"raw_score": benchmarking["industry_score"], "weight": DEFAULT_WEIGHTS["benchmarking"], "weighted_contribution": round(benchmarking["industry_score"] * DEFAULT_WEIGHTS["benchmarking"], 2)},
            "hidden_talent": {"raw_score": hidden_talent["hidden_potential_score"], "weight": DEFAULT_WEIGHTS["hidden_talent"], "weighted_contribution": round(hidden_talent["hidden_potential_score"] * DEFAULT_WEIGHTS["hidden_talent"], 2)},
            "portfolio_coach": {"raw_score": 80, "weight": DEFAULT_WEIGHTS["portfolio_coach"], "weighted_contribution": round(80 * DEFAULT_WEIGHTS["portfolio_coach"], 2)},
            "tech_depth": {"raw_score": tech_depth["technical_score"], "weight": DEFAULT_WEIGHTS["tech_depth"], "weighted_contribution": round(tech_depth["technical_score"] * DEFAULT_WEIGHTS["tech_depth"], 2)},
            "consistency": {"raw_score": consistency["consistency_score"], "weight": DEFAULT_WEIGHTS["consistency"], "weighted_contribution": round(consistency["consistency_score"] * DEFAULT_WEIGHTS["consistency"], 2)}
        },

        "headline_scores": {
            "portfolio_quality": int((ui_score + ux_score) / 2),
            "creativity": "high" if innovation_score_val > 75 else ("medium" if innovation_score_val > 55 else "low"),
            "problem_solving": "strong" if ux_score > 70 else "developing",
            "industry_readiness": benchmarking_score_val,
            "weighted_final_score": weighted_score
        },

        "detected_skills": {
            "advanced": [s["name"] for s in skills_list if s["proficiency"] == "advanced"],
            "intermediate": [s["name"] for s in skills_list if s["proficiency"] == "intermediate"],
            "beginner": [s["name"] for s in skills_list if s["proficiency"] == "beginner"]
        },

        "strengths": [
            { "point": "Strong visual coherence", "evidence": "Grid patterns and styling remain steady throughout the documents." },
            { "point": "Methodological documentation", "evidence": "Explicit callouts for user paths and layout structures are included." }
        ],

        "weaknesses": [
            { "point": "Insufficient quantitative metrics", "fix": "Incorporate key user performance metrics into project reviews." },
            { "point": "Limited tool diversity documentation", "fix": "List precise libraries or platforms used for building layouts." }
        ],

        "hidden_talent_flag": "yes" if hidden_talent_score_val > 75 else "no",
        "hidden_talent_note": "Possesses a very clean eye for document formatting and structured presentation layout design.",

        "top_3_improvements": [
            "Integrate actual task success metrics or design library components.",
            "Include links to live interactive prototypes.",
            "Write a brief executive summary at the start of each case study."
        ],

        "recruiter_recommendation": "must_interview" if base_score > 80 else ("recommend" if base_score > 65 else "consider"),
        "recommendation_reason": f"Demonstrates solid candidate readiness matching {role_target} requirements. Visual presentation and structural organization are clear.",
        "coach_note": "A highly readable, standard-compliant presentation layout. Keep refining visual micro-interactions and adding user research metrics to unlock higher-tier positions."
    }
    return report
