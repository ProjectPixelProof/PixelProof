"""Gold-blind closed-choice prompt construction and answer scoring."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import unicodedata
from pathlib import Path

from . import PROMPT_VERSION
from .datasets import EvalExample

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "visual_question_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "One answer copied exactly from allowed_answers.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    },
}


def text_prompt(example: EvalExample) -> str:
    choices = "\n".join(f"- {item}" for item in example.candidates)
    return (
        "Answer the visual question using only the supplied image and question.\n"
        "Select exactly one answer from the allowed list. Do not use outside tools.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Allowed answers:\n{choices}\n\n"
        "Return the selected answer exactly as written and a confidence from 0 to 1."
    )


def image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"unsupported image MIME type for {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def messages(example: EvalExample, *, image_detail: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt(example)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(example.image_path),
                        "detail": image_detail,
                    },
                },
            ],
        }
    ]


def sanitized_request(example: EvalExample, *, image_detail: str, image_sha256: str) -> dict:
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": text_prompt(example),
        "image_path": str(example.image_path),
        "image_sha256": image_sha256,
        "image_detail": image_detail,
        "response_format": RESPONSE_FORMAT,
    }


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n\"'`.,:;!?")


def parse_response(content: str, candidates: tuple[str, ...]) -> dict:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("response content is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"answer", "confidence"}:
        raise ValueError("response must contain exactly answer and confidence")
    answer = payload["answer"]
    confidence = payload["confidence"]
    if not isinstance(answer, str):
        raise ValueError("response answer must be a string")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("response confidence must be numeric")
    if not 0 <= confidence <= 1:
        raise ValueError("response confidence must be in [0, 1]")
    normalized = {normalize_answer(candidate): candidate for candidate in candidates}
    selected = normalized.get(normalize_answer(answer))
    if selected is None:
        raise ValueError("response answer is outside the allowed candidates")
    return {"answer": selected, "confidence": float(confidence)}
