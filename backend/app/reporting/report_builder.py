from __future__ import annotations

from app.evaluation.models import EvaluationReport, ReportSummary, Severity, TechniqueRunResult
from app.scoring import highest_severity


def build_report(results: tuple[TechniqueRunResult, ...]) -> EvaluationReport:
    if not results:
        summary = ReportSummary(
            run_id="",
            target_id="",
            total_techniques=0,
            matched_detections=0,
            highest_severity=Severity.INFO,
            score=0,
        )
        return EvaluationReport(summary=summary, results=results)

    detections = tuple(det for result in results for det in result.detections)
    summary = ReportSummary(
        run_id=results[0].run_id,
        target_id=results[0].target_id,
        total_techniques=len(results),
        matched_detections=sum(1 for det in detections if det.matched),
        highest_severity=highest_severity(detections),
        score=sum(result.score for result in results),
    )
    return EvaluationReport(summary=summary, results=results)
