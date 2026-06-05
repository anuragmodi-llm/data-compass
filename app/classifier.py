import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, AutoModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip2-base-patch16-224"
CATEGORIES_PATH = Path(__file__).parent.parent / "categories.yaml"


# ── Device detection ──────────────────────────────────────────────────────────

def _detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Category tree ─────────────────────────────────────────────────────────────

@dataclass
class Category:
    label: str
    display_name: str
    hypotheses: list[str]           # always a list; single-prompt classes have len == 1


@dataclass
class CategoryTree:
    categories: list[Category]
    gate_type: str
    default_threshold: float
    secondary_multiplier: float
    reject_label: str


def _load_category_tree(path: Path = CATEGORIES_PATH) -> CategoryTree:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    gate = raw["gate"]
    categories: list[Category] = []
    for entry in raw["categories"]:
        if "hypotheses" in entry:
            hypotheses = entry["hypotheses"]
        else:
            hypotheses = [entry["prompt"]]
        categories.append(Category(
            label=entry["label"],
            display_name=entry["display_name"],
            hypotheses=hypotheses,
        ))

    return CategoryTree(
        categories=categories,
        gate_type=gate["type"],
        default_threshold=float(gate["default_threshold"]),
        secondary_multiplier=float(gate["secondary_multiplier"]),
        reject_label=gate["reject_label"],
    )


# ── Model + category state (module-level singletons) ─────────────────────────

_device: torch.device = _detect_device()
_model: AutoModel | None = None
_processor: AutoProcessor | None = None
_category_tree: CategoryTree | None = None
_tree_lock = threading.RLock()


def _ensure_model() -> tuple[AutoModel, AutoProcessor]:
    global _model, _processor
    if _model is None:
        logger.info("Loading SigLIP2 model on device=%s …", _device)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModel.from_pretrained(MODEL_ID).to(_device).eval()
        logger.info("SigLIP2 model loaded.")
    return _model, _processor


def get_category_tree() -> CategoryTree:
    global _category_tree
    with _tree_lock:
        if _category_tree is None:
            _category_tree = _load_category_tree()
        return _category_tree


# ── Watchdog: hot-reload categories.yaml ─────────────────────────────────────

class _CategoriesReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if Path(event.src_path).resolve() == CATEGORIES_PATH.resolve():
            global _category_tree
            try:
                new_tree = _load_category_tree()
                with _tree_lock:
                    _category_tree = new_tree
                logger.info("categories.yaml reloaded — %d categories active.", len(new_tree.categories))
            except Exception as exc:
                logger.error("Failed to reload categories.yaml: %s", exc)


def start_watcher() -> Observer:
    observer = Observer()
    observer.schedule(
        _CategoriesReloadHandler(),
        path=str(CATEGORIES_PATH.parent),
        recursive=False,
    )
    observer.daemon = True
    observer.start()
    logger.info("Watching %s for changes.", CATEGORIES_PATH)
    return observer


# ── Scoring primitives ────────────────────────────────────────────────────────

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@torch.inference_mode()
def _score_image_against_texts(
    image: Image.Image,
    texts: list[str],
    model: AutoModel,
    processor: AutoProcessor,
) -> list[float]:
    """
    Returns one sigmoid score per text, in the same order as `texts`.
    SigLIP produces logits_per_image shaped [1, num_texts]; we sigmoid each.
    """
    inputs = processor(
        text=texts,
        images=[image] * len(texts),
        return_tensors="pt",
        padding="max_length",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # logits_per_image: [num_images, num_texts] → here [len(texts), len(texts)]
    # We paired each text with a copy of the same image, so take the diagonal.
    logits = outputs.logits_per_image  # [len(texts), len(texts)]
    scores = _sigmoid(logits.diagonal()).tolist()
    return scores


def _score_page(
    image: Image.Image,
    tree: CategoryTree,
    model: AutoModel,
    processor: AutoProcessor,
) -> dict:
    """
    Score one PIL image against all categories.

    Returns:
        {
            "scores":      { label: float, ... },
            "winner":      label string (may be reject_label),
            "winner_score": float,
            "passed_gate": bool,
        }
    """
    threshold = tree.default_threshold
    scores: dict[str, float] = {}

    for cat in tree.categories:
        if len(cat.hypotheses) == 1:
            # Single-hypothesis: one forward pass, one score
            raw = _score_image_against_texts(image, cat.hypotheses, model, processor)
            scores[cat.label] = raw[0]
        else:
            # Multi-hypothesis: score all, take max (max-sigmoid aggregation)
            raw = _score_image_against_texts(image, cat.hypotheses, model, processor)
            scores[cat.label] = max(raw)

    # Gate: keep only classes that meet the threshold
    passing = {label: s for label, s in scores.items() if s >= threshold}

    if not passing:
        return {
            "scores": scores,
            "winner": tree.reject_label,
            "winner_score": scores.get(tree.reject_label, 0.0),
            "passed_gate": False,
        }

    winner = max(passing, key=lambda lbl: passing[lbl])
    return {
        "scores": scores,
        "winner": winner,
        "winner_score": passing[winner],
        "passed_gate": True,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def classify(preprocessed: dict) -> dict:
    """
    Classify a preprocessed document.

    Args:
        preprocessed: dict returned by preprocessor.preprocess()
            {
                "images":     [PIL.Image, ...],
                "page_count": int,
                "processor":  AutoProcessor,   # ignored — classifier owns the model
            }

    Returns:
        {
            "label":        str,
            "display_name": str,
            "score":        float,
            "passed_gate":  bool,
            "page_count":   int,
            "best_page":    int,          # 1-indexed; always 1 for single-page docs
            "pages":        [            # per-page breakdown
                {
                    "page":        int,
                    "winner":      str,
                    "score":       float,
                    "passed_gate": bool,
                    "scores":      { label: float, ... },
                },
                ...
            ],
        }
    """
    model, processor = _ensure_model()
    tree = get_category_tree()
    images: list[Image.Image] = preprocessed["images"]

    page_results = []
    for idx, image in enumerate(images):
        result = _score_page(image, tree, model, processor)
        page_results.append({
            "page": idx + 1,
            "winner": result["winner"],
            "score": result["winner_score"],
            "passed_gate": result["passed_gate"],
            "scores": result["scores"],
        })

    # Representative result: page with highest winning-class score
    best = max(page_results, key=lambda p: p["score"])

    # Resolve display_name for the winning label
    label_to_display = {cat.label: cat.display_name for cat in tree.categories}
    winning_label = best["winner"]
    display_name = label_to_display.get(winning_label, winning_label)

    return {
        "label": winning_label,
        "display_name": display_name,
        "score": best["score"],
        "passed_gate": best["passed_gate"],
        "page_count": preprocessed["page_count"],
        "best_page": best["page"],
        "pages": page_results,
    }
