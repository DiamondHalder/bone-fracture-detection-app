import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sys
import os

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model = YOLO(resource_path("best.pt"))



CONF_THRESHOLD = 0.2

CLASS_INFO = {
    "Comminuted":           {"desc": "Bone shattered into multiple fragments",         "color": "#E24B4A"},
    "Healthy":              {"desc": "No fracture detected",                            "color": "#639922"},
    "Linear":               {"desc": "Thin crack along the bone without displacement", "color": "#378ADD"},
    "Oblique Displaced":    {"desc": "Diagonal fracture with bone displacement",        "color": "#D85A30"},
    "Oblique":              {"desc": "Diagonal fracture without displacement",          "color": "#EF9F27"},
    "Segmental":            {"desc": "Two fracture lines isolating a bone segment",     "color": "#534AB7"},
    "Spiral":               {"desc": "Fracture twists around the bone shaft",          "color": "#1D9E75"},
    "Transverse Displaced": {"desc": "Horizontal fracture with bone displacement",     "color": "#D4537E"},
    "Transverse":           {"desc": "Horizontal fracture without displacement",       "color": "#185FA5"},
}

def run_detection(image):
    if image is None:
        return None, "<p style='color:#888;'>No image provided.</p>"

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False,
    )

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    names = results[0].names

    if boxes is None or len(boxes) == 0:
        info_html = _no_detection_html()
        return Image.fromarray(annotated_rgb), info_html

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = names[cls_id]
        detections.append({"label": label, "conf": conf})

    info_html = _build_result_html(detections)
    return Image.fromarray(annotated_rgb), info_html


def _no_detection_html():
    return """
<div style="padding:1rem 0;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.75rem;">
    <span style="width:10px;height:10px;border-radius:50%;background:#639922;display:inline-block;"></span>
    <span style="font-size:15px;font-weight:500;color:var(--color-text-primary);">No fracture detected</span>
  </div>
  <p style="font-size:13px;color:var(--color-text-secondary);margin:0;">
    No findings. The bone appears healthy or the image quality may be insufficient for analysis.
  </p>
</div>
"""


def _build_result_html(detections):
    rows = ""
    for d in detections:
        label = d["label"]
        conf  = d["conf"]
        info  = CLASS_INFO.get(label, {"desc": "", "color": "#888"})
        color = info["color"]
        desc  = info["desc"]
        pct   = f"{conf * 100:.1f}%"

        rows += f"""
<div style="
  border:0.5px solid var(--color-border-tertiary);
  border-radius:10px;
  padding:0.85rem 1rem;
  margin-bottom:10px;
  background:var(--color-background-primary);
">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0;"></span>
      <span style="font-size:14px;font-weight:500;color:var(--color-text-primary);">{label}</span>
    </div>
    <span style="font-size:13px;color:var(--color-text-secondary);">{pct}</span>
  </div>
  <div style="background:var(--color-border-tertiary);border-radius:4px;height:4px;margin-bottom:8px;">
    <div style="width:{pct};height:4px;border-radius:4px;background:{color};"></div>
  </div>
  <p style="font-size:12px;color:var(--color-text-secondary);margin:0;">{desc}</p>
</div>
"""

    count = len(detections)
    plural = "finding" if count == 1 else "findings"

    return f"""
<div style="padding:0.5rem 0 0;">
  <p style="font-size:12px;color:var(--color-text-secondary);margin:0 0 0.75rem;letter-spacing:0.04em;text-transform:uppercase;">
    {count} {plural} detected
  </p>
  {rows}
  <p style="font-size:11px;color:var(--color-text-secondary);margin-top:0.75rem;">
    Confidence threshold: 0.25 &nbsp;·&nbsp; Model: YOLO26m fine-tuned &nbsp;·&nbsp; For research use only
  </p>
</div>
"""


css = """
* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto;
}

#header {
    border-bottom: 1px solid #e5e7eb;
    padding: 1.5rem 0 1.25rem;
    margin-bottom: 1.75rem;
}

#header h1 {
    font-size: 22px;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 4px;
    letter-spacing: -0.02em;
}

#header p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}

#badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    color: #1e40af;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.upload-label {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    margin-bottom: 6px;
}

#run-btn {
    background: #0f172a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer;
    width: 100%;
    margin-top: 8px;
    transition: background 0.15s;
}

#run-btn:hover {
    background: #1e293b !important;
}

#output-img {
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    overflow: hidden;
}

#result-panel {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    background: #f8fafc;
    min-height: 200px;
}

#footer {
    border-top: 1px solid #e5e7eb;
    padding: 1rem 0 0;
    margin-top: 1.5rem;
    font-size: 11px;
    color: #94a3b8;
    text-align: center;
}
"""

with gr.Blocks(css=css, title="Bone Fracture Detection System") as demo:

    gr.HTML("""
    <div id="header">
      <div id="badge">Research Tool</div>
      <h1>Bone Fracture Detection System</h1>
      <p>Upload an X-ray image to detect and classify bone fractures using a fine-tuned YOLO26m model trained on 9 fracture classes.</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<p class="upload-label">X-ray image</p>')
            image_input = gr.Image(
                type="pil",
                label="",
                elem_id="input-img",
                height=340,
            )
            run_btn = gr.Button("Analyze", elem_id="run-btn")

        with gr.Column(scale=1):
            gr.HTML('<p class="upload-label">Detection output</p>')
            image_output = gr.Image(
                label="",
                elem_id="output-img",
                height=340,
            )

        with gr.Column(scale=1):
            gr.HTML('<p class="upload-label">Findings</p>')
            result_html = gr.HTML(
                value="<p style='font-size:13px;color:#94a3b8;padding:0.5rem 0;'>Upload an image and click Analyze.</p>",
                elem_id="result-panel",
            )

    run_btn.click(
        fn=run_detection,
        inputs=[image_input],
        outputs=[image_output, result_html],
    )

    gr.HTML("""
    <div id="footer">
      Bone Fracture Detection System &nbsp;·&nbsp; YOLOv26 &nbsp;·&nbsp; 9-class model &nbsp;·&nbsp; mAP@50: 96.60% (val) &nbsp;·&nbsp; For research use only. Not for clinical diagnosis.
    </div>
    """)

demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)