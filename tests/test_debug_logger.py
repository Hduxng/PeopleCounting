from pathlib import Path

from debug import DebugLogger


def test_gallery_skip_summary_is_written_on_close(tmp_path: Path):
    logger = DebugLogger(enabled=True, output_dir=str(tmp_path))
    logger.log_gallery_skip(frame_idx=10, tid=7, reason="low_conf", confidence=0.41, crop_area=1024)
    logger.log_gallery_skip(frame_idx=11, tid=7, reason="drift_hold", confidence=0.92, crop_area=2048)
    logger.log_gallery_skip(frame_idx=12, tid=8, reason="low_conf", confidence=0.38, crop_area=960)
    logger.close()

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1

    events_log = (run_dirs[0] / "events.log").read_text(encoding="utf-8", errors="ignore")
    assert "G_SKIP_SUMMARY total=3" in events_log
    assert "G_SKIP_BY_REASON low_conf=2  drift_hold=1" in events_log
    assert "G_SKIP_BY_TID tid=7  total=2  drift_hold=1  low_conf=1" in events_log
    assert "G_SKIP_BY_TID tid=8  total=1  low_conf=1" in events_log
