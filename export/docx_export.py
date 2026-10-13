"""DOCX Export — ATS-friendly Word document generator.

ATS-safe: standard single-column layout, standard margins, clean heading styles,
no text boxes, no headers/footers with critical text, pure semantic paragraphs and lists.
"""

import io
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def export_resume_to_docx(resume_data: dict) -> bytes:
    """Generate ATS-friendly DOCX resume bytes."""
    doc = Document()

    # 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    DARK_BLUE = RGBColor(30, 41, 59)
    GRAY = RGBColor(100, 116, 139)

    contact = resume_data.get("contact", {})
    name = contact.get("name", "Resume")

    # Name Title
    p_name = doc.add_paragraph()
    p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_name = p_name.add_run(name)
    run_name.font.name = "Calibri"
    run_name.font.size = Pt(20)
    run_name.font.bold = True
    run_name.font.color.rgb = DARK_BLUE

    # Contact line
    contact_parts = []
    if contact.get("email"):
        contact_parts.append(contact["email"])
    if contact.get("phone"):
        contact_parts.append(contact["phone"])
    if contact.get("location"):
        contact_parts.append(contact["location"])
    if contact.get("linkedin"):
        contact_parts.append(contact["linkedin"])
    if contact.get("github"):
        contact_parts.append(contact["github"])

    if contact_parts:
        p_contact = doc.add_paragraph()
        p_contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_contact = p_contact.add_run("  |  ".join(contact_parts))
        run_contact.font.name = "Calibri"
        run_contact.font.size = Pt(9.5)
        run_contact.font.color.rgb = GRAY

    def add_section_heading(title: str):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(title.upper())
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = DARK_BLUE

    # Summary
    summary = resume_data.get("summary", "")
    if summary:
        add_section_heading("Professional Summary")
        p_sum = doc.add_paragraph()
        p_sum.paragraph_format.space_after = Pt(6)
        run_sum = p_sum.add_run(summary)
        run_sum.font.name = "Calibri"
        run_sum.font.size = Pt(10)

    # Skills
    skills = resume_data.get("skills", {})
    if skills:
        add_section_heading("Technical Skills")
        for cat, items in skills.items():
            if items:
                p_skill = doc.add_paragraph()
                p_skill.paragraph_format.space_after = Pt(2)
                run_cat = p_skill.add_run(f"{cat.replace('_', ' ').title()}: ")
                run_cat.font.name = "Calibri"
                run_cat.font.size = Pt(10)
                run_cat.font.bold = True
                
                item_str = ", ".join(items) if isinstance(items, list) else str(items)
                run_items = p_skill.add_run(item_str)
                run_items.font.name = "Calibri"
                run_items.font.size = Pt(10)

    # Experience
    experience = resume_data.get("experience", [])
    if experience:
        add_section_heading("Professional Experience")
        for exp in experience:
            p_exp = doc.add_paragraph()
            p_exp.paragraph_format.space_before = Pt(6)
            p_exp.paragraph_format.space_after = Pt(2)
            
            role = exp.get("role", "")
            company = exp.get("company", "")
            loc = exp.get("location", "")
            dates = exp.get("dates", "")
            
            run_role = p_exp.add_run(role)
            run_role.font.name = "Calibri"
            run_role.font.size = Pt(10.5)
            run_role.font.bold = True
            
            run_comp = p_exp.add_run(f" — {company}")
            run_comp.font.name = "Calibri"
            run_comp.font.size = Pt(10)
            run_comp.font.italic = True
            
            meta = []
            if loc:
                meta.append(loc)
            if dates:
                meta.append(dates)
            if meta:
                run_meta = p_exp.add_run(f" ({', '.join(meta)})")
                run_meta.font.name = "Calibri"
                run_meta.font.size = Pt(9.5)
                run_meta.font.color.rgb = GRAY

            bullets = exp.get("bullets", [])
            for b in bullets:
                p_bullet = doc.add_paragraph(style="List Bullet")
                p_bullet.paragraph_format.space_after = Pt(1.5)
                p_bullet.paragraph_format.space_before = Pt(0)
                run_b = p_bullet.add_run(b)
                run_b.font.name = "Calibri"
                run_b.font.size = Pt(9.5)

    # Projects
    projects = resume_data.get("projects", [])
    if projects:
        add_section_heading("Projects")
        for proj in projects:
            p_proj = doc.add_paragraph()
            p_proj.paragraph_format.space_before = Pt(4)
            p_proj.paragraph_format.space_after = Pt(1)
            
            p_name_run = p_proj.add_run(proj.get("name", "Project"))
            p_name_run.font.name = "Calibri"
            p_name_run.font.size = Pt(10)
            p_name_run.font.bold = True
            
            tech = proj.get("technologies", [])
            if tech:
                tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
                run_t = p_proj.add_run(f" | {tech_str}")
                run_t.font.name = "Calibri"
                run_t.font.size = Pt(9.5)
                run_t.font.italic = True
            
            bullets = proj.get("bullets", [])
            for b in bullets:
                p_b = doc.add_paragraph(style="List Bullet")
                p_b.paragraph_format.space_after = Pt(1)
                run_b = p_b.add_run(b)
                run_b.font.name = "Calibri"
                run_b.font.size = Pt(9.5)

    # Education
    education = resume_data.get("education", [])
    if education:
        add_section_heading("Education")
        for edu in education:
            p_edu = doc.add_paragraph()
            p_edu.paragraph_format.space_after = Pt(2)
            degree = edu.get("degree", "")
            institution = edu.get("institution", "")
            dates = edu.get("dates", "")
            
            run_deg = p_edu.add_run(f"{degree}, {institution}")
            run_deg.font.name = "Calibri"
            run_deg.font.size = Pt(10)
            run_deg.font.bold = True
            
            if dates:
                run_d = p_edu.add_run(f" ({dates})")
                run_d.font.name = "Calibri"
                run_d.font.size = Pt(9.5)
                run_d.font.color.rgb = GRAY

    # Certifications
    certs = resume_data.get("certifications", [])
    if certs:
        add_section_heading("Certifications")
        for cert in certs:
            p_c = doc.add_paragraph(style="List Bullet")
            p_c.paragraph_format.space_after = Pt(1)
            c_text = cert if isinstance(cert, str) else cert.get("name", "")
            run_c = p_c.add_run(c_text)
            run_c.font.name = "Calibri"
            run_c.font.size = Pt(9.5)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
