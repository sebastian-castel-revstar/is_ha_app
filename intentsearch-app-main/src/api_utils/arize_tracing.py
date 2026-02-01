import json
import os
from functools import wraps

from arize.otel import register as _register
from aws_lambda_powertools.utilities.parameters import get_secret
from openinference.instrumentation.bedrock import BedrockInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

context_propagator = TraceContextTextMapPropagator()


def register(project_name, auto_instrument_bedrock=False, **kwargs):
    arize = json.loads(get_secret(os.environ["arize_secret"]))
    # Setup OpenTelemetry tracing
    default_args = dict(
        space_id=arize["SpaceID"],
        api_key=arize["APIKey"],
        log_to_console=False,
        set_global_tracer_provider=True,
        project_name=project_name,
    )
    default_args.update(kwargs)
    tracer_provider = _register(**default_args)

    if auto_instrument_bedrock:
        # Initialize Bedrock auto-instrumentation
        BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider


def otel_trace(
    span_kind=None,
    span_name=None,
    additional_attributes=None,
    auto_flush=False,
    inherit_context=False,
    set_io=True,
):
    """Wrapper to trace method

    Args:
        span_kind (str, optional): OpenInference span kind (use semantic conventions from openinference.semconv.trace.OpenInferenceSpanKindValues). Defaults to None.
        span_name (str, optional): Name to assign to span, if customization is desired. Defaults to the method name being decorated.
        additional_attributes (dict, optional): Attributes to assign to span. Defaults to None.
        auto_flush (bool, optional): Whether to force spans to be flushed on method completion.
                                     Useful for Lambda handlers, otherwise spans might not be written.
                                     Defaults to False.
        inherit_context (bool, optional): Used to assign span context to the new span.
                                          Used to propagate context between services, maintaining correct
                                          parent/child relationships. Defaults to False.
        set_io (bool, optional): Whether to use the method arguments and return values as attributes. Defaults to True.

    Returns:
        Callable: Wrapper which creates a span for a decorated method
    """
    tracer = trace.get_tracer(__name__)

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # can only inherit context from first argument, as in lambda_handler(event, context)
            context = (
                context_propagator.extract(carrier=args[0]) if inherit_context else None
            )
            nonlocal span_name
            with tracer.start_as_current_span(
                name=(span_name or func.__name__), context=context
            ) as span:
                if span_kind:
                    span.set_attribute(
                        SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind
                    )
                if additional_attributes:
                    for key, value in additional_attributes.items():
                        span.set_attribute(key, value)
                if set_io:
                    span.set_attribute(
                        SpanAttributes.INPUT_VALUE, str(args) + str(kwargs)
                    )

                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    if set_io:
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
                    return result
                except Exception as err:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise err
                finally:
                    if auto_flush:
                        tracer.span_processor.force_flush()

        return inner

    return wrapper


def set_attributes(**attrs):
    span = trace.get_current_span()
    span.set_attributes(attrs)
