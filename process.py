
import re

# ======================================================================
# 1.  mapping helper
# ======================================================================
def get_eval_group(subtype: str):
    """Map dataset subtype → evaluation group tag."""
    return {
        "viewfactor.scalar":                     "view_factor_scalar",
        "viewfactor.dominance":                  "view_factor_binary",
        "viewfactor.sparsity":                   "view_factor_binary",
        "presence.binary":                       "presence_binary",
        "count":                                 "count_numeric",
        "depth.range":                           "depth_binary",
        "depth.variation":                       "depth_binary",
        "depth.range.label":                     "depth_categorical",
        "depth.average_per_object":              "depth_numeric",
        "depth.closest_object":                  "depth_closest_object",
        "depth.foreground_presence":             "depth_binary",
        "depth.background_presence":             "depth_binary",
        "spatial.distribution":                  "layout_binary",
        "spatial.distribution.label":            "layout_text",
        "segmentation.vertical_layout":          "layout_binary",
        "scene.skyline_visibility":              "layout_binary",
        "segmentation.vertical_layout.entity":   "layout_top_entity",
        "object.cooccurrence":                   "cooccurrence_binary",
        "occlusion.binary":                      "occlusion_binary",
        "spatial.relation":                      "advanced_spatial",
    }.get(subtype,
          "advanced_negation"      if "negation"      in subtype else
          "advanced_multihop"      if "multi_hop"     in subtype else
          "advanced_counterfactual"if "counterfactual" in subtype else
          None)

# ======================================================================
# 1.5  object remapping rules
# ======================================================================
OBJECT_REMAPPING = {
    # Vehicle group
    "car": "vehicle", "cars": "vehicle", "bus": "vehicle", "buses": "vehicle",
    "motorcycle": "vehicle", "motorcycles": "vehicle", "train": "vehicle", "trains": "vehicle",
    "truck": "vehicle", "trucks": "vehicle", "van": "vehicle", "vans": "vehicle",
    "bicycle": "vehicle", "bicycles": "vehicle",

    # Greenery group
    "tree": "greenery", "trees": "greenery", "vegetation": "greenery", "forest": "greenery",
    "shrub": "greenery", "shrubs": "greenery", "plant": "greenery", "plants": "greenery",
    "grass": "greenery", "bush": "greenery", "bushes": "greenery", "leaf": "greenery", "leaves": "greenery",

    # Person group
    "person": "person", "people": "person", "persons": "person", "pedestrian": "person",
    "pedestrians": "person", "human": "person", "humans": "person", "man": "person", "men": "person",
    "woman": "person", "women": "person", "child": "person", "children": "person",

    # Building group
    "building": "building", "buildings": "building", "skyscraper": "building", "skyscrapers": "building",
    "structure": "building", "structures": "building", "tower": "building", "towers": "building",
    "house": "building", "houses": "building", "office": "building", "offices": "building",
}

def remap_object_name(obj_name: str) -> str:
    """Remap object names to standardized categories"""
    if not isinstance(obj_name, str):
        return obj_name

    obj_lower = obj_name.lower().strip()
    return OBJECT_REMAPPING.get(obj_lower, obj_lower)

# ======================================================================
# 2.  number extractor
# ======================================================================
def extract_number(text: str):
    if not isinstance(text, str): return None
    txt = text.lower().strip()

    if re.match(r'^(\d+\s+)\1{2,}$', txt):            # "3 3 3 3"
        match = re.search(r'\d+', txt)
        if match:
            return int(match.group())
        return None
    if txt.isdigit() and len(txt) == 1:               # single digit
        return int(txt)
    if "0 to 1" in txt or "0-1" in txt:
        return 0.5

    WORD2NUM = {
        "none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a couple": 2, "couple": 2, "a few": 3, "few": 3,
        "several": 4, "some": 4, "many": 6, "a lot": 8,
        "half a dozen": 6, "dozen": 12, "hundred": 100,
    }
    for phrase, val in WORD2NUM.items():
        if phrase in txt:
            return val

    m = re.search(r'[-+]?\d*\.\d+', txt)              # decimal first
    if m: return float(m.group())
    m = re.search(r'[-+]?\d+', txt)                   # then int
    if m: return int(m.group())
    return None

# ======================================================================
# 3.  debug helpers
# ======================================================================
DEBUG_PARSE = False
def _dbg(tag, value):
    if DEBUG_PARSE:
        print(f"[parse-dbg] {tag}: {repr(value)}")
    return value

# ======================================================================
# 4.  model-output cleaning
# ======================================================================
_LLAVA_CHAT_RE = re.compile(
    r'###\s*human:.*?###\s*assistant:\s*',
    flags=re.IGNORECASE | re.DOTALL,
)

def clean_model_output(text: str) -> str:
    """
    Normalise raw model output.

    1.  Keep only the assistant's final utterance when the string contains
        LLaVA / Alpaca style chat markers.
    2.  Strip markdown fences and whitespace.
    3.  Return the *bare* answer token whenever possible.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    matches = list(_LLAVA_CHAT_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():].lstrip()

    lower = text.lower()
    if lower.startswith("assistant:"):
        text = text[len("assistant:"):].lstrip()

    if re.fullmatch(r'\d+(\.\d+)?', text) or text.lower() in {"yes", "no"}:
        return text.lower()

    text = re.sub(r"^```.*?```$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^>+\s*", "", text).strip()
    text = re.sub(r'[*_`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

# ======================================================================
# 5.  answer parser
# ======================================================================
def parse_answer(ans: str, group: str, subtype: str | None = None, outlier_threshold: float = 1000):
    """Convert raw answer string to canonical form for metric-side compare."""
    BINARY_GROUPS = {
        "view_factor_binary", "presence_binary", "depth_binary",
        "layout_binary", "cooccurrence_binary", "occlusion_binary",
        "advanced_negation", "advanced_multihop", "advanced_counterfactual",
        "advanced_spatial",
    }

    # 5.1 missing / empty input
    if not isinstance(ans, str) or not ans.strip():
        if group in BINARY_GROUPS:
            return 0.0
        if group in {"count_numeric", "depth_numeric"}:
            return 0
        if group == "view_factor_scalar":
            return 0.0
        if group == "depth_categorical":
            return "unknown"
        if group == "layout_text":
            return "other"
        if group in {"depth_closest_object", "layout_top_entity"}:
            return "unknown"
        return None

    ans_clean = clean_model_output(ans)
    if not ans_clean:
        return 0.0 if group in BINARY_GROUPS else None

    # 5.2 special-case subtypes that need string matching
    if subtype in {"negation.exclusion_choice", "multi_hop.which_is_more"}:
        # Look for answer after "answer:" or "answer-" anywhere in the text
        # Use a more specific pattern to capture only the object name, not the reasoning
        m = re.search(r'answer[:\-]\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', ans_clean, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).strip().lower()
            # Apply object remapping
            return remap_object_name(answer)
        # Fallback: look for the last word that could be an object name
        words = ans_clean.split()
        for word in reversed(words):
            if word.lower() in {"tree", "car", "bench", "people", "cars", "person", "truck", "bus", "bicycle", "motorcycle", "stop sign", "traffic light", "fire hydrant", "potted plant", "train", "skateboard", "clock", "sheep"}:
                # Apply object remapping
                return remap_object_name(word.lower())
        return remap_object_name(ans_clean.strip())

    # 5.3 explicit 'answer:' or 'answer-' suffix for closest-object and layout_top_entity
    if group in {"depth_closest_object", "layout_top_entity"} or subtype in {"depth.closest_object", "segmentation.vertical_layout.entity"}:
        # Look for answer after "answer:" anywhere in the text (not just at end)
        # Use a more specific pattern to capture only the object name, not the reasoning
        m = re.search(r'answer[:\-]\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', ans_clean, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).strip().lower()
            # Clean up common prefixes/suffixes
            answer = re.sub(r'^(the|a|an)\s+', '', answer)
            answer = re.sub(r'\s+(is|are|was|were|appears|seems).*$', '', answer)
            # Apply object remapping
            return remap_object_name(answer)

        # Look for patterns like "the object closest to the camera is X" or "X is closest"
        m = re.search(r'(\b\w+(?:\s+\w+)*?)(?=\s+(?:is|was|appears|seems)\s+(?:closest|the closest))', ans_clean)
        if m:
            answer = m.group(1).strip().lower()
            # Clean up common prefixes
            answer = re.sub(r'^(the|a|an)\s+', '', answer)
            # Apply object remapping
            return remap_object_name(answer)

        # Look for "answer: X" pattern with word boundaries
        m = re.search(r'answer:\s*(\b\w+(?:\s+\w+)*\b)', ans_clean, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).strip().lower()
            # Apply object remapping
            return remap_object_name(answer)

        # Last resort: take the last meaningful word that's not a common verb/preposition
        common_words = {"looking", "i", "i'm", "the", "a", "an", "is", "are", "was", "were", "appears", "seems", "closest", "object", "camera", "to", "of", "in", "on", "at", "with", "by", "for", "from", "that", "this", "these", "those", "what", "which", "one", "light", "foreground", "background"}
        words = ans_clean.split()
        for word in reversed(words):
            if word.lower() not in common_words and len(word) > 2:
                # Apply object remapping
                return remap_object_name(word.lower())

        # If all else fails, return the first non-common word
        for word in words:
            if word.lower() not in common_words and len(word) > 2:
                # Apply object remapping
                return remap_object_name(word.lower())

        return "unknown"

    # 5.4 numeric scalar (view_factor_scalar)
    if group == "view_factor_scalar":
        # First try to extract decimal numbers
        m = re.search(r'[-+]?\d*\.\d+', ans_clean)
        if m:
            try:
                value = float(m.group())
                # Check for outliers
                if abs(value) > outlier_threshold:
                    return 0.0  # Return default for outliers
                return value
            except ValueError:
                pass

        # Then try integers
        m = re.search(r'[-+]?\d+', ans_clean)
        if m:
            try:
                num = int(m.group())
                # Check for outliers
                if abs(num) > outlier_threshold:
                    return 0.0  # Return default for outliers
                return float(num)
            except ValueError:
                pass

        # Look for "answer: X" pattern
        m = re.search(r'answer[:\-]\s*([\d\.]+)', ans_clean, flags=re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1))
                # Check for outliers
                if abs(value) > outlier_threshold:
                    return 0.0  # Return default for outliers
                return value
            except ValueError:
                pass

        return 0.0

    # 5.5 counts / numeric outputs
    if group in {"count_numeric", "depth_numeric"}:
        num = extract_number(ans_clean)
        if num is not None:
            # Check for outliers
            if abs(num) > outlier_threshold:
                return 0  # Return default for outliers
            return num
        if "no" in ans_clean or "none" in ans_clean:
            return 0
        return 0

    # 5.6 categorical depth
    if group == "depth_categorical":
        # First, look for "answer:" patterns (highest priority)
        m = re.search(r'answer[:\-]\s*(\w+)', ans_clean, flags=re.IGNORECASE)
        if m:
            answer_part = m.group(1).lower()
            if answer_part in ["low", "moderate", "high"]:
                return answer_part
            # Handle partial matches
            if answer_part.startswith("h"):
                return "high"
            if answer_part.startswith("m"):
                return "moderate"
            if answer_part.startswith("l"):
                return "low"

        # Handle truncated answers - look for partial matches and context clues
        if "labeled" in ans_clean and ("depth complexity is" in ans_clean):
            # This looks like a truncated answer that was cut off at "labeled"
            # Look for context clues to determine the intended answer
            if any(word in ans_clean for word in ["extensive", "substantial", "significant", "complex", "varied", "greater than 40"]):
                return "high"
            if any(word in ans_clean for word in ["moderate", "medium", "average", "some", "between 20 and 40"]):
                return "moderate"
            if any(word in ans_clean for word in ["simple", "flat", "minimal", "little", "less than 20"]):
                return "low"

        # Look for context clues that might indicate the intended answer
        if any(word in ans_clean for word in ["extensive", "substantial", "significant", "complex", "varied"]):
            return "high"
        if any(word in ans_clean for word in ["moderate", "medium", "average", "some"]):
            return "moderate"
        if any(word in ans_clean for word in ["simple", "flat", "minimal", "little"]):
            return "low"

        # Last resort: look for exact matches in the text (but this should be lower priority)
        for cat in ("low", "moderate", "high"):
            if cat in ans_clean:
                return cat

        return "unknown"

    # 5.7 layout text options
    if group == "layout_text":
        # First, look for "answer:" patterns (highest priority)
        m = re.search(r'answer[:\-]\s*(\w+(?:\s+\w+)*)', ans_clean, flags=re.IGNORECASE)
        if m:
            answer_text = m.group(1).lower().strip()
            # Handle "evenly spread" -> "even"
            if "evenly spread" in answer_text or "evenly" in answer_text:
                return "even"
            # Handle "left side" or just "left"
            if "left" in answer_text:
                return "left side"
            # Handle "right side" or just "right"
            if "right" in answer_text:
                return "right side"
            # Handle other variations
            if any(k in answer_text for k in ("even", "centre", "center", "both", "middle")):
                return "even"

        # Fallback to keyword matching in the text (lower priority)
        if "left" in ans_clean:
            return "left side"
        if "right" in ans_clean:
            return "right side"
        if any(k in ans_clean for k in ("even", "centre", "center", "both", "middle", "evenly")):
            return "even"
        return "other"

    # 5.8 binary groups
    if group in BINARY_GROUPS:
        # Look for explicit yes/no answers first
        yn = re.findall(r'\b(yes|no)\b', ans_clean)
        if yn:
            last = yn[-1]
            return 1.0 if last == 'yes' else 0.0

        # Look for "answer: yes/no" pattern
        m = re.search(r'answer[:\-]\s*(yes|no)', ans_clean, flags=re.IGNORECASE)
        if m:
            return 1.0 if m.group(1).lower() == 'yes' else 0.0

        # Look for positive/negative indicators
        if re.search(r'\b(true|correct|present|visible|contains)\b', ans_clean):
            return 1.0
        if re.search(r'\b(false|incorrect|absent|none|zero)\b', ans_clean):
            return 0.0

        # For negation subtypes, look for specific patterns
        if subtype and "negation" in subtype:
            # Look for "answer: X" where X is the excluded item
            m = re.search(r'answer[:\-]\s*(\w+(?:\s+\w+)*)', ans_clean, flags=re.IGNORECASE)
            if m:
                return remap_object_name(m.group(1).strip().lower())

        return 0.0

    # 5.9 default fallback
    return None

# ======================================================================
# 6.  equality check for metrics
# ======================================================================
def is_correct(pred, label, subtype: str | None = None):
    if subtype == "negation.exclusion_choice":
        return (isinstance(pred, str) and isinstance(label, str)
                and label.strip().lower() in pred.strip().lower())
    if subtype == "multi_hop.which_is_more":
        return (isinstance(pred, str) and isinstance(label, str)
                and label.strip().lower() in pred.strip().lower())
    if isinstance(pred, float) and isinstance(label, float):
        return abs(pred - label) < 1e-4
    if isinstance(pred, (int, float)) and isinstance(label, (int, float)):
        return int(pred) == int(label)
    return str(pred).strip().lower() == str(label).strip().lower()
