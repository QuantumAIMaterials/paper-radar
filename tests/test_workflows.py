from pathlib import Path


def test_quantum_ai_workflow_is_isolated_from_2d_radar() -> None:
    project_root = Path(__file__).resolve().parent.parent
    workflow = (project_root / ".github/workflows/quantum-ai-watch.yml").read_text(
        encoding="utf-8"
    )

    assert "config/quantum_ai_keywords.yml" in workflow
    assert "config/quantum_ai_recommender_prompt.txt" in workflow
    assert "state/quantum_ai_seen.json" in workflow
    assert "secrets.QUANTUM_AI_FEISHU_WEBHOOK_URL" in workflow
    assert "secrets.QUANTUM_AI_FEISHU_SIGNING_SECRET" in workflow
    assert 'cron: "53 0 * * *"' in workflow
    assert 'cron: "53 4 * * *"' in workflow
    assert 'cron: "38 5 * * *"' in workflow
    assert 'cron: "23 8 * * *"' in workflow
    assert "group: paper-radar-quantum-ai-${{" in workflow
    assert "inputs.dry_run && github.run_id || 'state'" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "--start-not-before 18:40" in workflow
    assert "--max-start-wait-minutes 330" in workflow
    assert "--deliver-not-before 19:08" in workflow
    assert "secrets.OPENALEX_API_KEY" in workflow
    assert "--once-per-beijing-day" in workflow
    assert "inputs.dry_run && github.ref_name || 'main'" in workflow
    assert "always() && !(github.event_name == 'workflow_dispatch'" in workflow
    assert "inputs.dry_run || inputs.resend_latest" in workflow
    assert "git pull --rebase origin main" in workflow
    assert "for attempt in 1 2 3" in workflow


def test_2d_workflow_updates_state_from_latest_main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    workflow = (project_root / ".github/workflows/paper-watch.yml").read_text(
        encoding="utf-8"
    )

    assert "inputs.dry_run && github.ref_name || 'main'" in workflow
    assert "always() && !(github.event_name == 'workflow_dispatch'" in workflow
    assert "inputs.dry_run || inputs.resend_latest" in workflow
    assert "git pull --rebase origin main" in workflow
    assert 'cron: "47 0 * * *"' in workflow
    assert 'cron: "47 4 * * *"' in workflow
    assert 'cron: "32 5 * * *"' in workflow
    assert 'cron: "17 8 * * *"' in workflow
    assert "group: paper-radar-2d-${{" in workflow
    assert "inputs.dry_run && github.run_id || 'state'" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "--start-not-before 18:30" in workflow
    assert "--max-start-wait-minutes 330" in workflow
    assert "--deliver-not-before 19:00" in workflow
    assert "secrets.OPENALEX_API_KEY" in workflow
    assert "--once-per-beijing-day" in workflow
    assert "for attempt in 1 2 3" in workflow
