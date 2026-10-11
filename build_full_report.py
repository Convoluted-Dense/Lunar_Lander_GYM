"""Academic & Technical Project Report Generator for FOAI Lunar Lander RL.

Reads template HTML directly (preserving all raw LaTeX equations verbatim)
Embeds base64 assets and renders PDF using Microsoft Edge headless.
"""

import os
import base64
import subprocess
import pymupdf

def get_base64_img(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return ""
    with open(filepath, "rb") as f:
        data = f.read()
    ext = os.path.splitext(filepath)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"

# Gather Assets
logo_b64 = get_base64_img("reports/woxsen_logo_stacked_clean.png")
plot_b64 = get_base64_img("reports/training_comparison.png")
adv_frame_b64 = get_base64_img("reports/rollouts/advanced_frame.png")
base_frame_b64 = get_base64_img("reports/rollouts/baseline_frame.png")

template_path = "reports/report_template.html"
with open(template_path, "r", encoding="utf-8") as f:
    template = f.read()

html_content = (
    template.replace("###LOGO_B64###", logo_b64)
    .replace("###PLOT_B64###", plot_b64)
    .replace("###BASE_FRAME_B64###", base_frame_b64)
    .replace("###ADV_FRAME_B64###", adv_frame_b64)
)

html_path = "reports/FOAI_LunarLander_RL_Project_Report.html"
pdf_path = "reports/FOAI_LunarLander_RL_Project_Report.pdf"

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[OK] Saved HTML report to {html_path} ({len(html_content)} bytes)")

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_exe):
    edge_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

print(f"[RUN] Generating PDF with {edge_exe}...")
abs_html_path = os.path.abspath(html_path).replace("\\", "/")
abs_pdf_path = os.path.abspath(pdf_path)

cmd = [
    edge_exe,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=10000",
    "--no-pdf-header-footer",
    f"--print-to-pdf={abs_pdf_path}",
    f"file:///{abs_html_path}"
]

res = subprocess.run(cmd, capture_output=True)
print("Return code:", res.returncode)

if os.path.exists(pdf_path):
    size = os.path.getsize(pdf_path)
    print(f"[SUCCESS] PDF successfully created: {pdf_path} ({size} bytes)")
    doc = pymupdf.open(pdf_path)
    print(f"Total Pages: {len(doc)}")
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        pix_path = f"reports/report_page_{i+1}.png"
        pix.save(pix_path)
        print(f"Rendered Page {i+1} preview -> {pix_path}")
else:
    print("[ERROR] PDF generation failed!")
