# Python language processing utilities for Qythera training
import re
from typing import List, Dict

LANG = "python"

EXTRACTION_PATTERNS = {
    "function": r"(?:def|function|func|fn|fun)\s+(\w+)\s*\(",
    "class": r"(?:class|struct|interface|enum)\s+(\w+)",
    "import": r"(?:import|from|require|#include|use)\s+([\w.]+)",
}

def extract_features(code: str) -> Dict[str, List[str]]:
    features = {}
    for feat_type, pattern in EXTRACTION_PATTERNS.items():
        features[feat_type] = re.findall(pattern, code)
    return features

def validate_syntax(code: str) -> bool:
    return len(code.strip()) > 0

def format_for_training(code: str, docstring: str = "") -> Dict:
    return {"code": code, "language": LANG, "docstring": docstring, "features": extract_features(code)}

def filter_code_samples(samples: List[Dict], min_length: int = 50, max_length: int = 10000) -> List[Dict]:
    return [s for s in samples if min_length <= len(s.get("code", "")) <= max_length]
