from engine.latency_profiler import LatencyProfiler


def test_latency_profiler_basic():
    p = LatencyProfiler(report_interval=0.0)  # force immediate report
    p.start_frame()
    p.mark('vision')
    p.mark('gesture')
    p.mark('audio_queue')
    # Ensure last_report is in the past so end_frame returns a report
    p._last_report = 0.0
    report = p.end_frame()
    assert report is not None
    assert 'frame_total' in report
    assert 'vision' in report
    assert 'audio_queue' in report
    # Values should be numeric dicts
    assert isinstance(report['frame_total']['avg_ms'], float)

