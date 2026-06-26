-- Prebuilt analysis views for run-level reporting and downstream modeling.

CREATE OR REPLACE VIEW analysis_run_overview AS
SELECT
    run_id,
    run_kind,
    evaluation_id,
    mode,
    status,
    requested_by,
    technique_key,
    technique_name,
    technique_family,
    technique_surface,
    target_kind,
    target_model,
    attacker_model,
    success,
    score,
    asr,
    matched_detections,
    highest_severity,
    unsafe_intent,
    created_at,
    completed_at,
    EXTRACT(EPOCH FROM (completed_at - created_at)) AS duration_seconds
FROM evaluation_runs
ORDER BY completed_at DESC;


CREATE OR REPLACE VIEW analysis_model_summary AS
SELECT
    target_model,
    attacker_model,
    technique_surface,
    COUNT(*) AS run_count,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
    ROUND(AVG(score)::numeric, 2) AS avg_score,
    ROUND(AVG(asr)::numeric, 2) AS avg_asr,
    ROUND(AVG(matched_detections)::numeric, 2) AS avg_detection_count,
    MAX(completed_at) AS last_run_at
FROM evaluation_runs
WHERE run_kind = 'technique'
GROUP BY target_model, attacker_model, technique_surface
ORDER BY run_count DESC, avg_score DESC;


CREATE OR REPLACE VIEW analysis_technique_summary AS
SELECT
    technique_key,
    technique_name,
    technique_family,
    technique_surface,
    target_model,
    COUNT(*) AS run_count,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
    ROUND(AVG(score)::numeric, 2) AS avg_score,
    ROUND(AVG(asr)::numeric, 2) AS avg_asr,
    ROUND(AVG(matched_detections)::numeric, 2) AS avg_detection_count,
    MAX(highest_severity) AS max_recorded_severity,
    MAX(completed_at) AS last_run_at
FROM evaluation_runs
WHERE run_kind = 'technique'
GROUP BY technique_key, technique_name, technique_family, technique_surface, target_model
ORDER BY technique_key, target_model;


CREATE OR REPLACE VIEW analysis_detection_summary AS
SELECT
    r.target_model,
    r.technique_key,
    d.detection_type,
    d.severity,
    COUNT(*) AS detection_rows,
    SUM(CASE WHEN d.matched THEN 1 ELSE 0 END) AS matched_count,
    ROUND(AVG(CASE WHEN d.matched THEN 1 ELSE 0 END)::numeric, 4) AS match_rate
FROM evaluation_run_detections d
JOIN evaluation_runs r ON r.run_id = d.run_id
GROUP BY r.target_model, r.technique_key, d.detection_type, d.severity
ORDER BY matched_count DESC, detection_rows DESC;


CREATE OR REPLACE VIEW analysis_round_details AS
SELECT
    c.run_id,
    r.technique_key,
    r.target_model,
    c.entry_index,
    c.kind,
    c.payload ->> 'round_number' AS round_number,
    c.payload -> 'attacker' ->> 'input' AS attacker_input,
    c.payload -> 'attacker' ->> 'output' AS attacker_output,
    c.payload -> 'target' ->> 'prompt' AS target_prompt,
    c.payload -> 'target' ->> 'response' AS target_response,
    c.payload -> 'judge' ->> 'score' AS judge_score,
    c.payload -> 'judge' ->> 'rationale' AS judge_rationale,
    c.payload -> 'target' ->> 'poisoned' AS poisoned_response
FROM evaluation_run_conversations c
JOIN evaluation_runs r ON r.run_id = c.run_id
WHERE c.kind IN ('round', 'single_shot')
ORDER BY r.completed_at DESC, c.entry_index ASC;
