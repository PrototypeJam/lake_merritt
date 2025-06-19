import pathlib, json
from core.otel.ingester import OTelTraceIngester

def test_ingester_produces_items():
    raw = pathlib.Path("fixtures/manual_traces.json").read_text()
    items = OTelTraceIngester().ingest_str(raw)
    assert items, "No items produced"
    assert "otel_trace" in items[0].metadata
    assert items[0].metadata["duration_ms"] is not None