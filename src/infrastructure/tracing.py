import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from configs.settings import settings

logger = logging.getLogger(__name__)

_is_initialized = False

def init_tracing(service_name: str):
    """Initialize OpenTelemetry tracing globally."""
    global _is_initialized
    if _is_initialized:
        return
    
    if not settings.enable_tracing:
        logger.info(f"Tracing is disabled. Not initializing OTel for {service_name}.")
        return

    # Set up resource
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        "environment": settings.environment
    })

    # Create the tracer provider
    provider = TracerProvider(resource=resource)

    # Configure the OTLP Exporter
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True  # For local/internal communication
        )
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        logger.info(f"Configured OTLP Exporter pointing to {settings.otel_exporter_otlp_endpoint}")
    except Exception as e:
        logger.error(f"Failed to configure OTLP Exporter: {e}")
        # Fallback to console in development
        if settings.environment == "development":
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set global provider
    trace.set_tracer_provider(provider)
    _is_initialized = True
    logger.info(f"OpenTelemetry tracing initialized for {service_name}")

def get_tracer(name: str):
    return trace.get_tracer(name)
