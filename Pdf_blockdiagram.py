import os
import zipfile

PROJECT_NAME = "pdf_diagram_analyzer"
BASE_DIR = os.path.join(os.getcwd(), PROJECT_NAME)

FILES = {
    "main.py": '''#!/usr/bin/env python3
\"\"\"
PDF Diagram Analyzer - Main Entry Point
Offline, air-gapped system for extracting and analyzing diagrams from PDFs.
\"\"\"

import argparse
import logging
import sys
import os
from pathlib import Path

# Add modules path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.pdf_extractor import PDFExtractor
from modules.image_filter import ImageFilter
from modules.preprocess import ImagePreprocessor
from modules.ocr import OCRProcessor
from modules.detector import DiagramDetector
from modules.parser import GraphParser
from modules.llm import LocalLLMExplainer
from modules.report import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiagramAnalyzer:
    \"\"\"Main orchestrator for the diagram analysis pipeline.\"\"\"
    
    def __init__(self, output_dir="output", models_dir="models"):
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize modules
        self.extractor = PDFExtractor()
        self.filter = ImageFilter()
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRProcessor()
        self.detector = DiagramDetector()
        self.parser = GraphParser()
        self.llm = LocalLLMExplainer(model_name="mistral:7b")  # Change as needed
        self.report_gen = ReportGenerator(self.output_dir)
        
    def process_pdf(self, pdf_path):
        \"\"\"Process entire PDF and generate analysis reports.\"\"\"
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract images from PDF
        images_data = self.extractor.extract_images(pdf_path)
        if not images_data:
            logger.error("No images extracted from PDF")
            return None
        
        logger.info(f"Extracted {len(images_data)} candidate images")
        
        # Step 2: Filter diagram candidates
        diagram_candidates = self.filter.filter_diagrams(images_data)
        logger.info(f"Identified {len(diagram_candidates)} diagram images")
        
        if not diagram_candidates:
            logger.warning("No diagrams detected in PDF")
            return None
        
        # Step 3: Process each diagram
        all_diagram_analyses = []
        for idx, img_info in enumerate(diagram_candidates):
            logger.info(f"Processing diagram {idx+1}/{len(diagram_candidates)} - Page {img_info['page_num']}")
            
            try:
                # Preprocess image
                processed_img = self.preprocessor.process(img_info['image'])
                
                # OCR
                ocr_text = self.ocr.extract_text(processed_img)
                img_info['ocr_text'] = ocr_text
                
                # Detect nodes and edges
                detections = self.detector.detect(processed_img, ocr_text)
                
                # Parse into structured graph
                graph = self.parser.build_graph(detections, img_info)
                
                # Generate LLM explanation
                explanation = self.llm.explain_graph(graph)
                
                # Store results
                analysis = {
                    'diagram_id': idx + 1,
                    'page_number': img_info['page_num'],
                    'image_path': img_info['saved_path'],
                    'extracted_text': ocr_text,
                    'graph': graph,
                    'explanation': explanation,
                    'confidence': detections.get('confidence', 0.5)
                }
                all_diagram_analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Failed to process diagram {idx+1}: {str(e)}")
                continue
        
        # Step 4: Generate reports
        report_files = self.report_gen.generate(all_diagram_analyses, pdf_path)
        
        logger.info(f"Analysis complete. Reports saved to {self.output_dir}")
        return report_files

def main():
    parser = argparse.ArgumentParser(description="Analyze diagrams in PDF files (air-gapped)")
    parser.add_argument("--input", "-i", required=True, help="Input PDF file path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--model", "-m", default="mistral:7b", help="Ollama model name")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    analyzer = DiagramAnalyzer(output_dir=args.output)
    analyzer.llm.model_name = args.model
    result = analyzer.process_pdf(args.input)
    
    if result:
        logger.info(f"Analysis successful. Results in {args.output}")
    else:
        logger.error("Analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
''',

    "modules/__init__.py": "# Empty init file to mark as package\n",

    "modules/pdf_extractor.py": '''\"\"\"PDF image extraction module using PyMuPDF.\"\"\"

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFExtractor:
    \"\"\"Extract all images from PDF (embedded and scanned).\"\"\"
    
    def __init__(self, temp_dir="temp_images"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def extract_images(self, pdf_path):
        \"\"\"Extract images from PDF, return list of dicts with image data and metadata.\"\"\"
        images_data = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Method 1: Extract embedded images
            image_list = page.get_images(full=True)
            
            if image_list:
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    img_path = self.temp_dir / f"page_{page_num+1}_img_{img_idx}.{image_ext}"
                    cv2.imwrite(str(img_path), cv_img)
                    
                    images_data.append({
                        'page_num': page_num + 1,
                        'image_id': img_idx,
                        'image': cv_img,
                        'saved_path': str(img_path),
                        'source': 'embedded'
                    })
            else:
                # Method 2: For scanned PDFs, render page as image
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_bytes))
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                img_path = self.temp_dir / f"page_{page_num+1}_scanned.png"
                cv2.imwrite(str(img_path), cv_img)
                
                images_data.append({
                    'page_num': page_num + 1,
                    'image_id': 0,
                    'image': cv_img,
                    'saved_path': str(img_path),
                    'source': 'scanned_page'
                })
        
        doc.close()
        logger.info(f"Extracted {len(images_data)} images from PDF")
        return images_data
''',

    "modules/image_filter.py": '''\"\"\"Filter images to identify diagrams using heuristics.\"\"\"

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageFilter:
    \"\"\"Identify diagram images from general images.\"\"\"
    
    def __init__(self, edge_threshold=0.05, min_contour_area=500):
        self.edge_threshold = edge_threshold  # Minimum edge density ratio
        self.min_contour_area = min_contour_area
        
    def filter_diagrams(self, images_data):
        \"\"\"Return only images that are likely diagrams.\"\"\"
        filtered = []
        for img_info in images_data:
            if self._is_diagram(img_info['image']):
                filtered.append(img_info)
                logger.debug(f"Image page {img_info['page_num']} classified as diagram")
            else:
                logger.debug(f"Image page {img_info['page_num']} rejected as non-diagram")
        return filtered
    
    def _is_diagram(self, image):
        \"\"\"Heuristic: edge density, contour count, rectangular shapes.\"\"\"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Count rectangular contours (approx polygons with 4 vertices)
        rect_count = 0
        for cnt in large_contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                rect_count += 1
        
        # Diagram if: high edge density, multiple rectangles, or many large contours
        is_diagram = (edge_density > self.edge_threshold and rect_count >= 2) or len(large_contours) > 3
        
        # Also check for text presence via simple variance (high variance suggests text)
        if not is_diagram:
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 100:  # High variance often means text/diagram
                is_diagram = True
                
        return is_diagram
''',

    "modules/preprocess.py": '''\"\"\"Image preprocessing for better diagram analysis.\"\"\"

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    \"\"\"Enhance diagram images for OCR and detection.\"\"\"
    
    def __init__(self):
        pass
    
    def process(self, image):
        \"\"\"Apply full preprocessing pipeline.\"\"\"
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive thresholding to handle uneven lighting
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Morphological cleaning
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Optionally resize if too small
        h, w = cleaned.shape
        if h < 500 or w < 500:
            scale = max(800/h, 800/w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            cleaned = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
''',

    "modules/ocr.py": '''\"\"\"OCR module using Tesseract (offline).\"\"\"

import pytesseract
import cv2
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    \"\"\"Extract text from diagram images.\"\"\"
    
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Configuration for diagram text (numbers, letters, symbols)
        self.config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._=+()'
        
    def extract_text(self, image):
        \"\"\"Extract all text from preprocessed image.\"\"\"
        try:
            text = pytesseract.image_to_string(image, config=self.config)
            cleaned = ' '.join(text.split())  # Normalize whitespace
            logger.debug(f"OCR extracted: {cleaned[:100]}...")
            return cleaned
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def extract_text_with_boxes(self, image):
        \"\"\"Extract text with bounding boxes for later mapping to blocks.\"\"\"
        try:
            data = pytesseract.image_to_data(image, config=self.config, output_type=pytesseract.Output.DICT)
            boxes = []
            for i, word in enumerate(data['text']):
                if word.strip():
                    boxes.append({
                        'text': word,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    })
            return boxes
        except Exception as e:
            logger.error(f"OCR box extraction failed: {str(e)}")
            return []
''',

    "modules/detector.py": '''\"\"\"Detect blocks, arrows, connectors using OpenCV heuristics (no ML required).\"\"\"

import cv2
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class DiagramDetector:
    \"\"\"Detect diagram components: blocks (rectangles), arrows (lines with arrowheads), connectors.\"\"\"
    
    def __init__(self):
        self.min_block_area = 200
        self.max_block_area = 50000
        self.arrow_angle_thresh = 30  # degrees
        
    def detect(self, image, ocr_text):
        \"\"\"Main detection pipeline.\"\"\"
        height, width = image.shape
        # Find all contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = self._detect_blocks(contours, width, height)
        arrows, connectors = self._detect_arrows_and_connectors(image, contours)
        
        # Map OCR text to blocks (simplified: assign text to closest block)
        blocks = self._assign_text_to_blocks(blocks, ocr_text)
        
        confidence = self._calculate_confidence(blocks, arrows)
        
        return {
            'nodes': blocks,
            'edges': arrows + connectors,
            'confidence': confidence
        }
    
    def _detect_blocks(self, contours, img_w, img_h):
        \"\"\"Detect rectangular blocks.\"\"\"
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_block_area or area > self.max_block_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:  # Quadrilateral
                x, y, w, h = cv2.boundingRect(cnt)
                # Avoid noise: aspect ratio reasonable
                if 0.2 < (w/h) < 5.0 and w > 20 and h > 15:
                    blocks.append({
                        'id': len(blocks),
                        'type': 'block',
                        'bbox': [x, y, w, h],
                        'text': '',
                        'center': (x + w//2, y + h//2)
                    })
        return blocks
    
    def _detect_arrows_and_connectors(self, image, contours):
        \"\"\"Detect lines and classify as arrows or connectors.\"\"\"
        # Use Hough Line Transform
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        arrows = []
        connectors = []
        
        if lines is None:
            return arrows, connectors
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Check for arrowhead at endpoint by looking for small triangle contours
            has_arrowhead = self._check_arrowhead(image, (x2, y2), angle)
            
            edge = {
                'id': len(arrows) + len(connectors),
                'from_node': None,  # Will be linked later
                'to_node': None,
                'type': 'arrow' if has_arrowhead else 'connector',
                'start': (x1, y1),
                'end': (x2, y2),
                'angle': angle
            }
            if has_arrowhead:
                arrows.append(edge)
            else:
                connectors.append(edge)
        
        return arrows, connectors
    
    def _check_arrowhead(self, image, tip, angle):
        \"\"\"Check if there's a triangular arrowhead near the tip.\"\"\"
        # Simple: look for a small contour near tip within 10px
        tip_x, tip_y = tip
        h, w = image.shape
        roi_size = 15
        x_start = max(0, tip_x - roi_size)
        y_start = max(0, tip_y - roi_size)
        x_end = min(w, tip_x + roi_size)
        y_end = min(h, tip_y + roi_size)
        
        if x_end <= x_start or y_end <= y_start:
            return False
        
        roi = image[y_start:y_end, x_start:x_end]
        contours_roi, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_roi:
            area = cv2.contourArea(cnt)
            if 10 < area < 200:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                if len(approx) == 3:  # Triangle
                    return True
        return False
    
    def _assign_text_to_blocks(self, blocks, ocr_text):
        \"\"\"Assign OCR text to nearest block based on position.\"\"\"
        # This is a simplified placeholder. Full implementation would use OCR boxes.
        # For now, assign entire OCR text to the largest block.
        if blocks and ocr_text:
            blocks[0]['text'] = ocr_text
        return blocks
    
    def _calculate_confidence(self, blocks, arrows):
        \"\"\"Calculate confidence score based on detection density.\"\"\"
        total_components = len(blocks) + len(arrows)
        if total_components == 0:
            return 0.1
        # Simple heuristic: more components = higher confidence
        confidence = min(0.95, total_components / 20.0)
        return max(0.3, confidence)
''',

    "modules/parser.py": '''\"\"\"Convert detections to structured graph JSON.\"\"\"

import logging
import math

logger = logging.getLogger(__name__)

class GraphParser:
    \"\"\"Build graph representation from detected nodes and edges.\"\"\"
    
    def build_graph(self, detections, img_info):
        \"\"\"Create JSON graph with node-edge topology.\"\"\"
        nodes = detections['nodes']
        edges = detections['edges']
        
        # Link edges to nearest nodes based on Euclidean distance
        for edge in edges:
            start_pt = edge['start']
            end_pt = edge['end']
            
            # Find closest node to start
            min_dist_start = float('inf')
            closest_start = None
            for node in nodes:
                dist = math.hypot(start_pt[0] - node['center'][0], start_pt[1] - node['center'][1])
                if dist < min_dist_start:
                    min_dist_start = dist
                    closest_start = node['id']
            
            # Find closest node to end
            min_dist_end = float('inf')
            closest_end = None
            for node in nodes:
                dist = math.hypot(end_pt[0] - node['center'][0], end_pt[1] - node['center'][1])
                if dist < min_dist_end:
                    min_dist_end = dist
                    closest_end = node['id']
            
            if closest_start is not None and closest_end is not None:
                edge['from_node'] = closest_start
                edge['to_node'] = closest_end
        
        # Build graph structure
        graph = {
            'diagram_id': img_info.get('page_num', 0),
            'source_page': img_info['page_num'],
            'image_path': img_info['saved_path'],
            'nodes': [
                {
                    'id': n['id'],
                    'label': n['text'] if n['text'] else f"Block_{n['id']}",
                    'type': n['type'],
                    'bbox': n['bbox']
                } for n in nodes
            ],
            'edges': [
                {
                    'id': e['id'],
                    'from': e['from_node'],
                    'to': e['to_node'],
                    'type': e['type'],
                    'direction': 'forward' if e['type'] == 'arrow' else 'bidirectional'
                } for e in edges if e['from_node'] is not None and e['to_node'] is not None
            ],
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'confidence': detections['confidence']
            }
        }
        
        return graph
''',

    "modules/llm.py": '''\"\"\"Local LLM module using Ollama (offline).\"\"\"

import json
import requests
import logging

logger = logging.getLogger(__name__)

class LocalLLMExplainer:
    \"\"\"Generate engineering explanations using local LLM.\"\"\"
    
    def __init__(self, model_name="mistral:7b", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._check_ollama()
    
    def _check_ollama(self):
        \"\"\"Verify Ollama is running and model is available.\"\"\"
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                logger.warning("Ollama not running. LLM explanations will be fallback.")
                self.available = False
                return
            models = response.json().get('models', [])
            if not any(m['name'].startswith(self.model_name.split(':')[0]) for m in models):
                logger.warning(f"Model {self.model_name} not found. Pull it first: ollama pull {self.model_name}")
                self.available = False
            else:
                self.available = True
        except:
            logger.warning("Cannot connect to Ollama. Ensure it's running.")
            self.available = False
    
    def explain_graph(self, graph):
        \"\"\"Send structured graph to LLM and return explanation.\"\"\"
        if not self.available:
            return self._fallback_explanation(graph)
        
        prompt = self._build_prompt(graph)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "max_tokens": 500}
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return self._parse_response(result['response'])
            else:
                logger.error(f"LLM error: {response.status_code}")
                return self._fallback_explanation(graph)
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            return self._fallback_explanation(graph)
    
    def _build_prompt(self, graph):
        \"\"\"Construct prompt for LLM.\"\"\"
        nodes_desc = "\\n".join([f"- Node {n['id']}: {n['label']}" for n in graph['nodes']])
        edges_desc = "\\n".join([f"- Edge from {e['from']} to {e['to']} ({e['type']})" for e in graph['edges']])
        
        prompt = f\"\"\"You are an expert systems engineer. Analyze the following diagram structure and provide:
1. System type (e.g., data flow, control system, block diagram)
2. Step-by-step flow explanation
3. Real-world applications

Diagram nodes:
{nodes_desc}

Connections:
{edges_desc}

Provide a concise, professional engineering analysis.\"\"\"
        return prompt
    
    def _parse_response(self, response_text):
        \"\"\"Extract structured explanation from LLM response.\"\"\"
        lines = response_text.strip().split('\\n')
        explanation = {
            'system_type': 'Unknown',
            'step_by_step': '',
            'real_world_apps': '',
            'raw_response': response_text
        }
        
        # Simple parsing of common headings
        current_section = None
        for line in lines:
            lower = line.lower()
            if 'system type' in lower or 'type:' in lower:
                explanation['system_type'] = line.split(':', 1)[-1].strip()
                current_section = 'type'
            elif 'step-by-step' in lower or 'flow explanation' in lower:
                current_section = 'steps'
                explanation['step_by_step'] = ''
            elif 'real-world' in lower or 'applications' in lower:
                current_section = 'apps'
                explanation['real_world_apps'] = ''
            else:
                if current_section == 'steps':
                    explanation['step_by_step'] += line + '\\n'
                elif current_section == 'apps':
                    explanation['real_world_apps'] += line + '\\n'
        
        if not explanation['system_type']:
            explanation['system_type'] = "General block diagram"
        
        return explanation
    
    def _fallback_explanation(self, graph):
        \"\"\"Provide rule-based explanation when LLM unavailable.\"\"\"
        num_nodes = graph['metadata']['total_nodes']
        num_edges = graph['metadata']['total_edges']
        
        return {
            'system_type': 'Unknown (LLM offline)',
            'step_by_step': f"The diagram contains {num_nodes} components and {num_edges} connections. Based on the structure, it appears to represent a flow or process. Each node likely represents a function or state, and arrows indicate direction of data or control flow.",
            'real_world_apps': 'Possible applications include software architecture, process automation, control systems, or data pipelines.',
            'raw_response': 'LLM not available - using fallback.'
        }
''',

    "modules/report.py": '''\"\"\"Generate final reports in JSON, text, and markdown formats.\"\"\"

import json
import os
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    \"\"\"Generate comprehensive analysis reports.\"\"\"
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        
    def generate(self, analyses, input_pdf_path):
        \"\"\"Generate all report formats.\"\"\"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(input_pdf_path).stem
        
        # JSON report
        json_path = self.output_dir / f"{base_name}_analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        logger.info(f"JSON report saved: {json_path}")
        
        # Human-readable text report
        txt_path = self.output_dir / f"{base_name}_report_{timestamp}.txt"
        self._generate_text_report(analyses, txt_path)
        
        # Markdown report
        md_path = self.output_dir / f"{base_name}_README_{timestamp}.md"
        self._generate_markdown_report(analyses, md_path)
        
        return {'json': str(json_path), 'txt': str(txt_path), 'md': str(md_path)}
    
    def _generate_text_report(self, analyses, filepath):
        \"\"\"Generate plain text report.\"\"\"
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\\n")
            f.write("PDF DIAGRAM ANALYSIS REPORT\\n")
            f.write(f"Generated: {datetime.now()}\\n")
            f.write("=" * 80 + "\\n\\n")
            
            for analysis in analyses:
                f.write(f"\\n--- DIAGRAM {analysis['diagram_id']} (Page {analysis['page_number']}) ---\\n")
                f.write(f"Confidence: {analysis['confidence']:.2f}\\n")
                f.write(f"Extracted Text:\\n{analysis['extracted_text'][:500]}\\n\\n")
                f.write("Detected Components (Nodes):\\n")
                for node in analysis['graph']['nodes']:
                    f.write(f"  - {node['label']}\\n")
                f.write("\\nConnections (Edges):\\n")
                for edge in analysis['graph']['edges']:
                    f.write(f"  - {edge['from']} → {edge['to']} ({edge['type']})\\n")
                f.write("\\nSystem Type: " + analysis['explanation'].get('system_type', 'N/A') + "\\n")
                f.write("\\nStep-by-step Explanation:\\n" + analysis['explanation'].get('step_by_step', 'N/A') + "\\n")
                f.write("\\nReal-world Applications:\\n" + analysis['explanation'].get('real_world_apps', 'N/A') + "\\n")
                f.write("-" * 40 + "\\n")
    
    def _generate_markdown_report(self, analyses, filepath):
        \"\"\"Generate markdown report.\"\"\"
        with open(filepath, 'w') as f:
            f.write(f"# PDF Diagram Analysis Report\\n\\n")
            f.write(f"**Generated:** {datetime.now()}\\n\\n")
            f.write(f"**Total Diagrams Found:** {len(analyses)}\\n\\n")
            
            for analysis in analyses:
                f.write(f"## Diagram {analysis['diagram_id']} (Page {analysis['page_number']})\\n")
                f.write(f"**Confidence:** {analysis['confidence']:.2f}\\n\\n")
                f.write("### Extracted Text\\n")
                f.write(f"```\\n{analysis['extracted_text'][:300]}\\n```\\n\\n")
                f.write("### Graph Structure\\n")
                f.write("#### Nodes\\n")
                for node in analysis['graph']['nodes']:
                    f.write(f"- **{node['label']}**\\n")
                f.write("\\n#### Edges\\n")
                for edge in analysis['graph']['edges']:
                    f.write(f"- `{edge['from']}` → `{edge['to']}` ({edge['type']})\\n")
                f.write("\\n### Engineering Explanation\\n")
                f.write(f"**System Type:** {analysis['explanation'].get('system_type', 'N/A')}\\n\\n")
                f.write(f"**Step-by-step Flow:**\\n{analysis['explanation'].get('step_by_step', 'N/A')}\\n\\n")
                f.write(f"**Real-world Applications:**\\n{analysis['explanation'].get('real_world_apps', 'N/A')}\\n\\n")
                f.write("---\\n\\n")
''',

    "requirements.txt": '''PyMuPDF==1.24.0
opencv-python==4.9.0.80
pytesseract==0.3.10
Pillow==10.1.0
numpy==1.26.2
ollama==0.1.8
requests==2.31.0
''',

    "README.md": '''# PDF Diagram Analyzer - Air-Gapped AI Agent

Complete offline system to extract diagrams from PDFs, convert to structured graphs, and generate engineering explanations using local LLM.

## Features

- ✅ Extract images from PDF (embedded and scanned)
- ✅ Identify diagram images using heuristics
- ✅ Preprocess images for OCR and detection
- ✅ OCR text extraction (Tesseract)
- ✅ Detect blocks, arrows, connectors using OpenCV
- ✅ Build graph representation (nodes + edges)
- ✅ Generate explanations via local LLM (Ollama)
- ✅ Multi-diagram support per page  
- ✅ Output JSON, text, and markdown reports

## Installation (Offline / Air-Gapped)

### Prerequisites

1. **Python 3.8+** with pip
2. **Tesseract OCR** binary:
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
   - Windows: Download installer from GitHub UB-Mannheim/tesseract
   - Add to PATH or set `tesseract_cmd` in code
3. **Ollama** (for LLM):
   - Download from https://ollama.com/download
   - Install offline: transfer `.deb` or `.exe` to target machine
   - Pull model (once online, then copy model files):
     ```bash
     ollama pull mistral:7b
     # Model stored in ~/.ollama/models - can be preloaded
     ```

### Python Dependencies

Create a local wheelhouse on an online machine:

```bash
pip download -r requirements.txt -d ./offline_packages
