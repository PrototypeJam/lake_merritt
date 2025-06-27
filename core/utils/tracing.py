# core/utils/tracing.py

class NoOpSpan:
    def set_attribute(self, key, value):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()

_tracer = NoOpTracer()

def get_tracer(name: str):
    """
    Returns a tracer instance. In this basic implementation, it's always a no-op tracer.
    A real implementation would check for OpenTelemetry SDK configuration.
    """
    return _tracer