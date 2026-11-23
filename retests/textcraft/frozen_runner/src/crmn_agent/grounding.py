"""Deterministic observation anchors: quotations, not an LLM's unverified interpretation."""
import re

from .schemas import Anchor


def observation_anchors(observation: str, mission: str, limit: int = 4) -> list[Anchor]:
    words = {w for w in re.findall(r'[a-z]+', mission.lower()) if len(w) > 3}
    chunks = []
    for match in re.finditer(r'[^\n.!?]+[.!?]?', observation):
        text = match.group().strip()
        while text:
            if len(text) <= 110:
                chunk, text = text, ''
            else:
                end = text.rfind(' ', 0, 110)
                end = end if end > 0 else 110
                chunk, text = text[:end], text[end:].lstrip()
            if chunk:
                score = len(words & set(re.findall(r'[a-z]+', chunk.lower())))
                chunks.append((score, len(chunks), chunk))
    selected = sorted(chunks, key=lambda item: (-item[0], item[1]))[:limit]
    return [Anchor(key=f'observed_{index}', value=text, evidence=text)
            for _, index, text in selected]

