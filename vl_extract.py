"""
vl_extract.py — Vision-Language Image Extraction for HRS documents
Extracts images from PDF/DOCX, sends them to llama.cpp VL model,
gets text descriptions to enrich the RAG context.

Usage:
  from vl_extract import extract_and_describe_images

  descriptions = extract_and_describe_images(
      filepath="hrs_document.pdf",
      llamacpp_url="http://localhost:8081"  # VL model on separate port
  )
"""

import os
import re
import base64
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import requests


# ============================================================
# IMAGE EXTRACTION FROM DOCUMENTS
# ============================================================

def extract_images_from_pdf(filepath, min_size_kb=5, max_images=10):
    """Extract images from PDF pages, skip tiny icons"""
    doc = fitz.open(filepath)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Method 1: Extract embedded images
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK or other
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")
                size_kb = len(img_bytes) / 1024

                if size_kb >= min_size_kb:
                    images.append({
                        "data": img_bytes,
                        "page": page_num + 1,
                        "type": "embedded",
                        "size_kb": round(size_kb, 1),
                        "width": pix.width,
                        "height": pix.height,
                    })
                pix = None
            except Exception as e:
                print(f"  [VL] Skip image page {page_num+1}: {e}")
                continue

        # Method 2: Render full page as image (catches diagrams drawn as vectors)
        # Only do this for pages that have few or no embedded images
        if not image_list:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            if len(img_bytes) / 1024 > 20:  # Non-trivial page
                images.append({
                    "data": img_bytes,
                    "page": page_num + 1,
                    "type": "page_render",
                    "size_kb": round(len(img_bytes) / 1024, 1),
                    "width": pix.width,
                    "height": pix.height,
                })

        if len(images) >= max_images:
            break

    doc.close()
    return images


def extract_images_from_docx(filepath, min_size_kb=5, max_images=10):
    """Extract images from DOCX files"""
    from docx import Document as DocxDocument

    doc = DocxDocument(filepath)
    images = []

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_bytes = rel.target_part.blob
                size_kb = len(img_bytes) / 1024

                if size_kb >= min_size_kb:
                    # Determine format
                    content_type = rel.target_part.content_type
                    ext = "png" if "png" in content_type else "jpg"

                    images.append({
                        "data": img_bytes,
                        "page": 0,
                        "type": f"docx_{ext}",
                        "size_kb": round(size_kb, 1),
                    })
            except Exception as e:
                print(f"  [VL] Skip DOCX image: {e}")

        if len(images) >= max_images:
            break

    return images


def extract_images(filepath, min_size_kb=5, max_images=10):
    """Extract images from any supported document"""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_images_from_pdf(filepath, min_size_kb, max_images)
    elif ext == ".docx":
        return extract_images_from_docx(filepath, min_size_kb, max_images)
    return []


# ============================================================
# VL MODEL QUERY (via llama.cpp with LLaVA/MiniCPM-V)
# ============================================================

def describe_image_with_vl(image_bytes, llamacpp_url, prompt=None):
    """
    Send image to VL model running on llama.cpp and get description.

    The VL model must be running on a separate llama.cpp instance:
      ./build/bin/llama-llava-cli or llama-server with multimodal model
    """
    if prompt is None:
        prompt = (
            "Analyze this technical diagram/figure from an engineering document. "
            "Describe: 1) What type of diagram is this (block diagram, flowchart, "
            "schematic, table, etc.) 2) Key components and their relationships "
            "3) Any labeled signals, interfaces, or data flows "
            "4) Any specifications, values, or parameters shown. "
            "Be concise and technical."
        )

    # Encode image to base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # llama.cpp multimodal API format
    payload = {
        "prompt": f"[img-1]\n{prompt}",
        "image_data": [{"data": img_b64, "id": 1}],
        "temperature": 0.1,
        "n_predict": 512,
        "stop": ["</s>"],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{llamacpp_url}/completion",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("content", "")
    except requests.exceptions.ConnectionError:
        return None  # VL model not running — skip silently
    except Exception as e:
        print(f"  [VL] Error: {e}")
        return None


# ============================================================
# MAIN FUNCTION: Extract + Describe
# ============================================================

def extract_and_describe_images(filepath, llamacpp_url="http://localhost:8081",
                                 min_size_kb=5, max_images=6):
    """
    Full pipeline: extract images from document, describe each with VL model.
    Returns list of {page, description, type, size_kb}

    If VL model is not running, returns empty list (graceful fallback).
    """
    print(f"[VL] Extracting images from {Path(filepath).name}...")
    images = extract_images(filepath, min_size_kb, max_images)
    print(f"[VL] Found {len(images)} images")

    if not images:
        return []

    # Check if VL model is available
    try:
        resp = requests.get(f"{llamacpp_url}/health", timeout=3)
        if resp.status_code != 200:
            print("[VL] VL model not running — skipping image analysis")
            return []
    except:
        print("[VL] VL model not available — skipping image analysis")
        return []

    descriptions = []
    for i, img in enumerate(images):
        print(f"[VL] Analyzing image {i+1}/{len(images)} "
              f"(page {img['page']}, {img['size_kb']}KB)...")

        desc = describe_image_with_vl(img["data"], llamacpp_url)
        if desc:
            descriptions.append({
                "page": img["page"],
                "description": desc,
                "type": img["type"],
                "size_kb": img["size_kb"],
            })
            print(f"[VL] ✓ Got description ({len(desc)} chars)")

    print(f"[VL] Completed: {len(descriptions)}/{len(images)} images described")
    return descriptions


def format_vl_descriptions_for_prompt(descriptions):
    """Format VL descriptions for inclusion in LLM prompt"""
    if not descriptions:
        return ""

    output = ["\n### DIAGRAM/IMAGE ANALYSIS (from VL model):"]
    for d in descriptions:
        output.append(f"\n[Page {d['page']} — {d['type']}]")
        output.append(d["description"])

    return "\n".join(output)
