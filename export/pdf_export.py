"""PDF Export — WeasyPrint HTML→PDF rendering.

ATS-safe: single-column, standard fonts, no tables/graphics that ATS chokes on.
"""

import os
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_template() -> str:
    """Load the HTML resume template."""
    path = os.path.join(TEMPLATE_DIR, "template.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_css() -> str:
    """Load the CSS for the resume template."""
    path = os.path.join(TEMPLATE_DIR, "template.css")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_resume_html(resume_data: dict) -> str:
    """Render structured resume data into HTML string."""
    template_str = _get_template()
    css = _get_css()

    template = Template(template_str)
    html = template.render(resume=resume_data, css=css)
    return html


def export_resume_to_pdf(resume_data: dict) -> bytes:
    """Export structured resume to PDF bytes via WeasyPrint."""
    try:
        from weasyprint import HTML
        html_string = render_resume_html(resume_data)
        pdf_bytes = HTML(string=html_string).write_pdf()
        return pdf_bytes
    except ImportError:
        logger.error("WeasyPrint not installed. Install with: pip install weasyprint")
        raise
    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        raise


def export_cover_letter_to_pdf(cover_letter_text: str, role_title: str = "", company: str = "") -> bytes:
    """Export cover letter text to PDF bytes."""
    try:
        from weasyprint import HTML

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
    body {{ font-family: 'Georgia', 'Times New Roman', serif; font-size: 11pt;
           line-height: 1.6; max-width: 700px; margin: 40px auto; padding: 40px;
           color: #1a1a1a; }}
    .header {{ margin-bottom: 30px; }}
    .header h2 {{ font-size: 14pt; margin: 0; color: #2c3e50; }}
    .header p {{ font-size: 10pt; color: #666; margin: 4px 0; }}
    .body {{ white-space: pre-wrap; }}
</style></head><body>
<div class="header">
    <h2>Cover Letter{f' — {role_title}' if role_title else ''}</h2>
    {f'<p>{company}</p>' if company else ''}
</div>
<div class="body">{cover_letter_text}</div>
</body></html>"""

        return HTML(string=html).write_pdf()
    except ImportError:
        raise ImportError("WeasyPrint not installed.")
