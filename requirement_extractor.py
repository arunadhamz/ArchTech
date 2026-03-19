"""
requirement_extractor.py — Multi-Layer Requirement Extraction Engine

Goal: Extract 100% of requirements from Tech Spec / HRS documents
Strategy: 3-layer approach (Rule-Based → LLM-Assisted → Gap Detection)

Usage:
    from requirement_extractor import extract_all_requirements
    
    results = extract_all_requirements(
        filepath="tech_spec.docx",
        llm_query_fn=query_llm,        # Your LLM function
        use_llm=True,                    # Enable Layer 2
    )
"""

import re
import json
from pathlib import Path
from collections import defaultdict


# ============================================================
# LAYER 1: RULE-BASED EXTRACTION (Fast, ~60-70% recall)
# ============================================================

# Pattern library — catches most common requirement formats
REQUIREMENT_PATTERNS = [
    # === EXPLICIT ID PATTERNS ===
    # REQ-001, HRS-042, SRS-FUNC-003, FR-01, NFR-12, etc.
    {
        "name": "explicit_id",
        "pattern": re.compile(
            r'(?P<id>(?:REQ|HRS|SRS|SDD|SYRS|SR|HR|FR|NFR|IR|PR|SAF|CON|SEC|ENV|IF|PERF)[-_.]?\d{1,5}(?:\.\d{1,3})?)'
            r'\s*[:\-–—]\s*(?P<text>.{15,})',
            re.IGNORECASE
        ),
        "confidence": 0.95,
    },

    # === "SHALL" LANGUAGE ===
    # "The system shall...", "The software must...", "The module will..."
    {
        "name": "shall_language",
        "pattern": re.compile(
            r'(?P<text>'
            r'(?:The\s+)?'
            r'(?:system|software|hardware|firmware|module|component|interface|unit|device|'
            r'processor|controller|FPGA|board|platform|subsystem|driver|application|server|client)'
            r'\s+'
            r'(?:shall|must|should|will|needs?\s+to|is\s+required\s+to|has\s+to)'
            r'\s+.{15,})',
            re.IGNORECASE
        ),
        "confidence": 0.90,
    },

    # === NUMBERED REQUIREMENTS IN LISTS ===
    # "1.2.3 The processor shall...", "3.4.1) Support..."
    {
        "name": "numbered_req",
        "pattern": re.compile(
            r'^\s*(?P<id>\d{1,3}(?:\.\d{1,3}){1,4})\s*[.):\-]?\s*'
            r'(?P<text>(?:The\s+)?(?:system|software|hardware|shall|must|should|support|provide|enable|ensure|implement|handle|process).{15,})',
            re.IGNORECASE | re.MULTILINE
        ),
        "confidence": 0.85,
    },

    # === TABLE ROW PATTERN ===
    # "| REQ-001 | Description | Priority |"
    {
        "name": "table_row",
        "pattern": re.compile(
            r'\|\s*(?P<id>[A-Z]{2,5}[-_.]?\d{1,5})\s*\|\s*(?P<text>[^|]{15,}?)\s*\|',
            re.IGNORECASE
        ),
        "confidence": 0.92,
    },

    # === BULLET WITH REQUIREMENT VERBS ===
    # "- Process data at 100MHz", "* Support MIL-STD-1553B"
    {
        "name": "bullet_requirement",
        "pattern": re.compile(
            r'^\s*[-•*▪►]\s*(?P<text>'
            r'(?:shall|must|should|will|provide|support|enable|ensure|maintain|implement|'
            r'handle|manage|control|monitor|detect|process|compute|generate|transmit|'
            r'receive|store|display|validate|verify|authenticate|interface|comply|conform|'
            r'operate|withstand|tolerate|respond|accept|reject|log|record|report)'
            r'\s+.{10,})',
            re.IGNORECASE | re.MULTILINE
        ),
        "confidence": 0.80,
    },

    # === CONSTRAINT LANGUAGE ===
    # "Operating temperature: -40°C to +85°C", "Power consumption shall not exceed..."
    {
        "name": "constraint",
        "pattern": re.compile(
            r'(?P<text>'
            r'(?:operating\s+temperature|humidity|altitude|vibration|shock|weight|dimension|'
            r'power\s+consumption|voltage|current|frequency|bandwidth|latency|throughput|'
            r'response\s+time|mtbf|mttr|availability|reliability|ip\s*rating|'
            r'emc|electromagnetic|certification|compliance|ingress|'
            r'temperature\s+range|storage\s+temperature)'
            r'\s*[:]\s*.{10,})',
            re.IGNORECASE
        ),
        "confidence": 0.85,
    },

    # === PERFORMANCE VALUES ===
    # "within 100ms", "at least 99.9%", "minimum 10Gbps"
    {
        "name": "performance_value",
        "pattern": re.compile(
            r'(?P<text>.{10,}?'
            r'(?:within\s+\d|at\s+least\s+\d|minimum\s+\d|maximum\s+\d|'
            r'no\s+(?:more|less|greater|fewer)\s+than\s+\d|'
            r'not\s+exceed\s+\d|up\s+to\s+\d|greater\s+than\s+\d|less\s+than\s+\d)'
            r'.{5,})',
            re.IGNORECASE
        ),
        "confidence": 0.75,
    },

    # === INTERFACE DEFINITIONS ===
    # "Interface with...", "Communicate via...", "Connect to..."
    {
        "name": "interface_def",
        "pattern": re.compile(
            r'(?P<text>'
            r'(?:interface|communicate|connect|interoperate|integrate|interact)\s+'
            r'(?:with|via|through|over|using|to)\s+.{10,})',
            re.IGNORECASE
        ),
        "confidence": 0.78,
    },
]

# === REQUIREMENT TYPE CLASSIFICATION ===
TYPE_KEYWORDS = {
    "functional": {
        "keywords": [
            "shall", "process", "compute", "calculate", "generate", "display",
            "store", "retrieve", "transmit", "receive", "execute", "perform",
            "provide", "support", "enable", "handle", "manage", "control",
            "monitor", "detect", "identify", "classify", "validate", "verify",
            "authenticate", "encrypt", "decode", "convert", "filter", "route",
            "log", "record", "report", "notify", "alert", "trigger",
        ],
        "weight": 1.0,
    },
    "performance": {
        "keywords": [
            "latency", "throughput", "bandwidth", "response time", "within",
            "millisecond", "microsecond", "per second", "fps", "mbps", "gbps",
            "mhz", "ghz", "capacity", "maximum", "minimum", "rate", "speed",
            "concurrent", "real-time", "real time", "deadline", "uptime",
            "availability", "99.9", "mtbf", "mttr", "scalab",
        ],
        "weight": 1.5,  # Higher weight — fewer keywords but more distinctive
    },
    "interface": {
        "keywords": [
            "interface", "protocol", "api", "uart", "spi", "i2c", "pcie", "usb",
            "ethernet", "can bus", "mil-std", "arinc", "rs-232", "rs-485", "gpio",
            "hdmi", "lvds", "jtag", "tcp", "udp", "http", "mqtt", "modbus",
            "connector", "pin", "port", "bus", "link", "socket", "handshake",
            "packet", "frame", "message format", "data format", "signal",
        ],
        "weight": 1.5,
    },
    "safety": {
        "keywords": [
            "safety", "fail-safe", "failsafe", "redundan", "fault", "error handling",
            "watchdog", "timeout", "recovery", "backup", "integrity", "checksum",
            "crc", "parity", "ecc", "radiation", "critical", "hazard", "risk",
            "protection", "isolation", "emergency", "shutdown", "graceful",
            "do-178", "do-254", "iec 61508", "iso 26262", "sil", "dal", "asil",
        ],
        "weight": 2.0,  # Highest weight — safety is critical to identify
    },
    "non_functional": {
        "keywords": [
            "maintainab", "reliab", "availab", "portab", "usab", "testab",
            "scalab", "extensib", "reusab", "modular", "configurable",
            "documentation", "training", "user manual", "backward compatible",
        ],
        "weight": 1.0,
    },
    "constraint": {
        "keywords": [
            "operating temperature", "humidity", "altitude", "vibration", "shock",
            "weight", "dimension", "power consumption", "voltage range", "current",
            "ip rating", "ingress", "environmental", "emc", "electromagnetic",
            "certification", "compliance", "rohs", "reach", "ce mark", "fcc",
            "mil-spec", "size", "form factor",
        ],
        "weight": 1.5,
    },
    "regulatory": {
        "keywords": [
            "do-178", "do-254", "mil-std", "mil-hdbk", "ieee", "iso",
            "iec", "rtca", "certification", "qualification", "airworthiness",
            "faa", "easa", "drdo", "bis", "stanag", "def-stan",
            "itar", "ear", "export control",
        ],
        "weight": 2.0,
    },
}


def classify_requirement_type(text):
    """Classify requirement type using weighted keyword matching"""
    text_lower = text.lower()
    scores = {}

    for req_type, config in TYPE_KEYWORDS.items():
        score = sum(
            config["weight"] for kw in config["keywords"] if kw in text_lower
        )
        scores[req_type] = score

    if max(scores.values(), default=0) == 0:
        return "functional"  # Default

    return max(scores, key=scores.get)


def extract_layer1_rules(text, source_info=""):
    """
    LAYER 1: Rule-based extraction using regex patterns.
    Fast, deterministic, catches ~60-70% of requirements.
    """
    requirements = []
    seen_texts = set()
    auto_id_counter = 1

    lines = text.split("\n")

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 15:
            continue

        for pat_config in REQUIREMENT_PATTERNS:
            match = pat_config["pattern"].search(stripped)
            if match:
                groups = match.groupdict()
                req_text = groups.get("text", stripped).strip()
                req_id = groups.get("id", None)

                # Clean up text
                req_text = re.sub(r'\s+', ' ', req_text).strip()
                if len(req_text) < 15:
                    continue

                # De-duplicate by text similarity
                text_key = req_text[:80].lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # Auto-assign ID if not found
                if not req_id or len(req_id) < 2:
                    req_id = f"AUTO-{auto_id_counter:04d}"
                    auto_id_counter += 1
                else:
                    req_id = req_id.upper()

                # Classify
                req_type = classify_requirement_type(req_text)

                requirements.append({
                    "id": req_id,
                    "text": req_text,
                    "type": req_type,
                    "extraction_method": f"rule:{pat_config['name']}",
                    "confidence": pat_config["confidence"],
                    "source_line": line_num + 1,
                    "source_section": source_info,
                    "original_line": stripped,
                })
                break  # One match per line

    return requirements


# ============================================================
# LAYER 2: LLM-POWERED EXTRACTION (Slow, catches 20-25% more)
# ============================================================

LLM_EXTRACTION_PROMPT = """You are a requirements engineer analyzing a technical document section.

Extract ALL requirements from the following text. A requirement is any statement that:
- Describes what the system SHALL/MUST/SHOULD do (functional)
- Specifies performance targets (speed, throughput, timing)
- Defines interfaces (protocols, connectors, data formats)
- States safety or reliability constraints
- Lists environmental or physical constraints
- Mentions compliance or certification needs
- Describes any constraint or capability the system must have

IMPORTANT: Extract requirements that are IMPLICIT or HIDDEN in descriptions too.
For example: "The radar processes signals at 100MHz" → implicit performance requirement.

For each requirement, output a JSON array:
[
  {{"text": "requirement text", "type": "functional|performance|interface|safety|constraint|regulatory|non_functional", "implicit": true/false}},
  ...
]

Output ONLY the JSON array, nothing else. No markdown, no explanation.

TEXT TO ANALYZE:
{text}
"""

LLM_EXTRACTION_PROMPT_SHORT = """Extract ALL requirements from this text as JSON array.
Include implicit/hidden requirements too.
Format: [{{"text": "...", "type": "functional|performance|interface|safety|constraint|regulatory", "implicit": false}}]
Output ONLY JSON array.

TEXT:
{text}
"""


def extract_layer2_llm(text, llm_query_fn, section_heading="", max_chunk_size=1500):
    """
    LAYER 2: LLM-powered extraction.
    Sends text sections to LLM to find requirements that regex missed.
    """
    if not llm_query_fn:
        return []

    requirements = []
    auto_id_counter = 5000  # Start from 5000 to avoid conflicts with Layer 1

    # Split text into manageable chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size // 5):  # ~5 chars per word
        chunk = " ".join(words[i:i + max_chunk_size // 5])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    for chunk_idx, chunk in enumerate(chunks):
        # Use shorter prompt for speed
        prompt = LLM_EXTRACTION_PROMPT_SHORT.format(text=chunk[:max_chunk_size])

        print(f"  [Layer 2] Analyzing chunk {chunk_idx+1}/{len(chunks)} "
              f"({len(chunk)} chars)...")

        result = llm_query_fn(prompt, system_prompt="You are a requirements extraction engine. Output only valid JSON.", temperature=0.1)

        if not result or result.startswith("ERROR"):
            print(f"  [Layer 2] LLM error on chunk {chunk_idx+1}")
            continue

        # Parse JSON response
        try:
            # Clean up common LLM output issues
            result = result.strip()
            # Remove markdown code fences
            result = re.sub(r'```json\s*', '', result)
            result = re.sub(r'```\s*', '', result)
            # Find the JSON array
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                continue

            for item in parsed:
                if isinstance(item, dict) and "text" in item:
                    req_text = item["text"].strip()
                    if len(req_text) < 15:
                        continue

                    req_type = item.get("type", "functional")
                    if req_type not in TYPE_KEYWORDS:
                        req_type = classify_requirement_type(req_text)

                    requirements.append({
                        "id": f"LLM-{auto_id_counter:04d}",
                        "text": req_text,
                        "type": req_type,
                        "extraction_method": "llm",
                        "confidence": 0.70,
                        "implicit": item.get("implicit", False),
                        "source_section": section_heading,
                    })
                    auto_id_counter += 1

        except (json.JSONDecodeError, Exception) as e:
            print(f"  [Layer 2] Parse error: {e}")
            continue

    return requirements


# ============================================================
# LAYER 3: GAP DETECTION (Finds sections with missing reqs)
# ============================================================

GAP_DETECTION_PROMPT = """You are a senior requirements engineer performing a gap analysis.

I have extracted {count} requirements from a document. Here are the document sections 
and how many requirements were found in each:

{section_summary}

Review this and identify:
1. Sections that likely contain requirements but have 0 or very few extracted
2. Types of requirements that may be missing (safety? performance? interface?)
3. Common requirement categories for this type of system that aren't represented

Output a JSON array of potential gaps:
[
  {{"section": "section name", "issue": "description of what might be missing", "priority": "high|medium|low"}},
  ...
]

Output ONLY the JSON array.
"""


def detect_gaps(sections_with_counts, total_reqs, llm_query_fn=None, extracted_types=None):
    """
    LAYER 3: Detect potential gaps in extraction.
    Returns warnings about sections that might have missed requirements.
    """
    gaps = []

    # Rule-based gap detection (no LLM needed)
    for section in sections_with_counts:
        heading = section["heading"].lower()
        count = section["req_count"]
        content_length = section["content_length"]

        # Sections with lots of content but no requirements → suspicious
        if content_length > 200 and count == 0:
            # Check if heading suggests requirements content
            req_headings = [
                "requirement", "specification", "function", "performance",
                "interface", "safety", "constraint", "capability",
                "feature", "description", "operation", "behavior",
                "design", "system", "hardware", "software",
            ]
            if any(kw in heading for kw in req_headings):
                gaps.append({
                    "section": section["heading"],
                    "issue": f"Section has {content_length} chars but 0 requirements extracted. "
                             f"This section heading suggests it should contain requirements.",
                    "priority": "high",
                    "content_preview": section.get("content_preview", "")[:200],
                })
            elif content_length > 500:
                gaps.append({
                    "section": section["heading"],
                    "issue": f"Large section ({content_length} chars) with 0 requirements. "
                             f"May contain implicit requirements.",
                    "priority": "medium",
                    "content_preview": section.get("content_preview", "")[:200],
                })

    # Check for missing requirement types
    if extracted_types:
        expected_types = {"functional", "performance", "interface", "safety", "constraint"}
        missing_types = expected_types - set(extracted_types.keys())
        for mt in missing_types:
            gaps.append({
                "section": "GLOBAL",
                "issue": f"No {mt} requirements found in entire document. "
                         f"Most Tech Specs / HRS documents have {mt} requirements.",
                "priority": "high" if mt == "safety" else "medium",
            })

    # LLM-powered gap detection (optional)
    if llm_query_fn and sections_with_counts:
        section_summary = "\n".join(
            f"  - {s['heading']}: {s['req_count']} requirements ({s['content_length']} chars)"
            for s in sections_with_counts
        )
        prompt = GAP_DETECTION_PROMPT.format(
            count=total_reqs,
            section_summary=section_summary
        )

        result = llm_query_fn(prompt, system_prompt="You are a gap analysis engine. Output only valid JSON.", temperature=0.1)
        if result and not result.startswith("ERROR"):
            try:
                result = re.sub(r'```json\s*', '', result.strip())
                result = re.sub(r'```\s*', '', result)
                match = re.search(r'\[.*\]', result, re.DOTALL)
                if match:
                    llm_gaps = json.loads(match.group())
                    for g in llm_gaps:
                        if isinstance(g, dict) and "section" in g:
                            g["detection_method"] = "llm"
                            gaps.append(g)
            except:
                pass

    return gaps


# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

def deduplicate_requirements(layer1_reqs, layer2_reqs):
    """
    Merge Layer 1 and Layer 2 results, removing duplicates.
    Layer 1 (rule-based) has priority for matching requirements.
    """
    merged = list(layer1_reqs)  # Start with all Layer 1
    layer1_texts = set()

    for req in layer1_reqs:
        # Create normalized key for matching
        normalized = re.sub(r'\s+', ' ', req["text"][:100].lower().strip())
        layer1_texts.add(normalized)

    # Add Layer 2 requirements that aren't duplicates
    added = 0
    skipped = 0
    for req in layer2_reqs:
        normalized = re.sub(r'\s+', ' ', req["text"][:100].lower().strip())

        # Check if similar text already exists
        is_duplicate = False
        for existing in layer1_texts:
            # Simple overlap check
            words_new = set(normalized.split())
            words_existing = set(existing.split())
            if len(words_new) > 0 and len(words_existing) > 0:
                overlap = len(words_new & words_existing) / min(len(words_new), len(words_existing))
                if overlap > 0.6:  # 60% word overlap = duplicate
                    is_duplicate = True
                    break

        if not is_duplicate:
            merged.append(req)
            layer1_texts.add(normalized)
            added += 1
        else:
            skipped += 1

    print(f"  [Dedup] Layer 2 added {added} new, skipped {skipped} duplicates")
    return merged


# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================

def extract_all_requirements(sections, llm_query_fn=None, use_llm=True):
    """
    Full 3-layer extraction pipeline.

    Args:
        sections: List of {"heading": str, "content": [str]} from document parser
        llm_query_fn: Function to query LLM (prompt, system_prompt, temperature) → str
        use_llm: Enable Layer 2 & 3 (LLM-assisted extraction)

    Returns:
        {
            "requirements": [...],       # All extracted requirements
            "gaps": [...],               # Potential missing requirements
            "stats": {...},              # Extraction statistics
            "sections_analysis": [...],  # Per-section breakdown
        }
    """
    print(f"\n{'='*60}")
    print(f"[EXTRACTION] Multi-Layer Requirement Extraction")
    print(f"[EXTRACTION] Sections: {len(sections)}")
    print(f"[EXTRACTION] LLM: {'enabled' if use_llm and llm_query_fn else 'disabled'}")
    print(f"{'='*60}")

    all_layer1 = []
    all_layer2 = []
    sections_analysis = []

    # ---- LAYER 1: Rule-Based ----
    print("\n[Layer 1] Rule-based extraction...")
    for section in sections:
        heading = section.get("heading", "Unknown")
        content = "\n".join(section.get("content", []))

        if not content.strip():
            continue

        reqs = extract_layer1_rules(content, source_info=heading)
        all_layer1.extend(reqs)

        sections_analysis.append({
            "heading": heading,
            "content_length": len(content),
            "content_preview": content[:200],
            "layer1_count": len(reqs),
            "req_count": len(reqs),  # Will be updated after Layer 2
        })

    print(f"[Layer 1] ✓ Found {len(all_layer1)} requirements")

    # ---- LAYER 2: LLM-Powered ----
    if use_llm and llm_query_fn:
        print("\n[Layer 2] LLM-assisted extraction...")

        # Only send sections that Layer 1 found few/no requirements in
        for i, section in enumerate(sections):
            heading = section.get("heading", "Unknown")
            content = "\n".join(section.get("content", []))

            if not content.strip() or len(content) < 100:
                continue

            # Find how many Layer 1 found in this section
            l1_count = sum(1 for r in all_layer1 if r.get("source_section") == heading)

            # If Layer 1 found very few relative to content size, use LLM
            content_per_req = len(content) / max(l1_count, 1)
            if l1_count == 0 or content_per_req > 500:
                print(f"  [Layer 2] Section: '{heading}' "
                      f"(L1 found {l1_count}, {len(content)} chars — needs LLM)")
                l2_reqs = extract_layer2_llm(content, llm_query_fn, heading)
                all_layer2.extend(l2_reqs)

                # Update section analysis
                for sa in sections_analysis:
                    if sa["heading"] == heading:
                        sa["layer2_count"] = len(l2_reqs)

        print(f"[Layer 2] ✓ Found {len(all_layer2)} additional requirements")
    else:
        print("\n[Layer 2] Skipped (LLM not available)")

    # ---- MERGE & DEDUPLICATE ----
    print("\n[Merge] Deduplicating...")
    all_requirements = deduplicate_requirements(all_layer1, all_layer2)

    # Re-number all requirements
    for i, req in enumerate(all_requirements):
        if req["id"].startswith("AUTO-") or req["id"].startswith("LLM-"):
            req_type_prefix = {
                "functional": "FUNC",
                "performance": "PERF",
                "interface": "IF",
                "safety": "SAF",
                "constraint": "CON",
                "regulatory": "REG",
                "non_functional": "NFR",
            }.get(req["type"], "REQ")
            req["id"] = f"EXT-{req_type_prefix}-{i+1:04d}"

    # Update section analysis with final counts
    for sa in sections_analysis:
        sa["req_count"] = sum(
            1 for r in all_requirements
            if r.get("source_section") == sa["heading"]
        )

    # ---- LAYER 3: GAP DETECTION ----
    print("\n[Layer 3] Gap detection...")
    type_counts = defaultdict(int)
    for r in all_requirements:
        type_counts[r["type"]] += 1

    gaps = detect_gaps(
        sections_analysis,
        len(all_requirements),
        llm_query_fn if use_llm else None,
        type_counts
    )
    print(f"[Layer 3] ✓ Found {len(gaps)} potential gaps")

    # ---- STATISTICS ----
    stats = {
        "total_requirements": len(all_requirements),
        "layer1_count": len(all_layer1),
        "layer2_count": len(all_layer2),
        "by_type": dict(type_counts),
        "by_method": {
            "rule_based": sum(1 for r in all_requirements if r["extraction_method"].startswith("rule:")),
            "llm_extracted": sum(1 for r in all_requirements if r["extraction_method"] == "llm"),
        },
        "implicit_count": sum(1 for r in all_requirements if r.get("implicit", False)),
        "high_confidence": sum(1 for r in all_requirements if r.get("confidence", 0) >= 0.85),
        "gaps_found": len(gaps),
        "high_priority_gaps": sum(1 for g in gaps if g.get("priority") == "high"),
    }

    print(f"\n{'='*60}")
    print(f"[EXTRACTION COMPLETE]")
    print(f"  Total requirements: {stats['total_requirements']}")
    print(f"  By method: Rule-based={stats['by_method']['rule_based']}, "
          f"LLM={stats['by_method']['llm_extracted']}")
    print(f"  By type: {dict(type_counts)}")
    print(f"  Implicit: {stats['implicit_count']}")
    print(f"  Gaps: {stats['gaps_found']} ({stats['high_priority_gaps']} high priority)")
    print(f"{'='*60}\n")

    return {
        "requirements": all_requirements,
        "gaps": gaps,
        "stats": stats,
        "sections_analysis": sections_analysis,
    }
