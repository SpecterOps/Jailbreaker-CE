from __future__ import annotations

from app.evaluation.models import DetectionResult, Severity


SEVERITY_SCORES: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 10,
    Severity.MEDIUM: 30,
    Severity.HIGH: 60,
    Severity.CRITICAL: 100,
}


def score_detections(detections: tuple[DetectionResult, ...]) -> int:
    return sum(SEVERITY_SCORES[item.severity] for item in detections if item.matched)


def highest_severity(detections: tuple[DetectionResult, ...]) -> Severity:
    ordered = sorted(detections, key=lambda item: SEVERITY_SCORES[item.severity], reverse=True)
    return ordered[0].severity if ordered else Severity.INFO
