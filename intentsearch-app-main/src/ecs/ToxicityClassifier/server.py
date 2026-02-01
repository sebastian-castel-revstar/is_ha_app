import os
import time
import traceback

from apiutils import instrumentation
from apiutils.cognito_auth import CognitoJWTAuthorizer
from aws_lambda_powertools import Logger
from ddtrace.contrib.asgi import TraceMiddleware
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Request,
    Response,
)
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from models import ErrorMessage
from models import Request as APIRequest
from models import Response as APIResponse
from toxicityAPI import lambda_handler

logger = Logger(log_uncaught_exceptions=True)


# https://fastapi.tiangolo.com/how-to/custom-request-and-route/#custom-apiroute-class-in-a-router
class TimedRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            start_time = time.perf_counter()
            response: Response = await original_route_handler(request)
            process_time = time.perf_counter() - start_time
            logger.info("Post request completed in %.2f ms", (1000 * process_time))
            return response

        return custom_route_handler


# healthcheck endpoint
app = FastAPI(root_path=".")


@app.middleware("http")
async def reroute_path(request: Request, call_next):
    # re-route requests from /toxicity-classifier/suffix to /suffix
    # ALB listener rules preserve full path when forwarding to ECS, but we only want /ping not /toxicity-classifier/ping
    base_path = request.scope["path"]
    if base_path.count("/") >= 2:
        request.scope["path"] = "/" + base_path.split("/", 2)[-1]
    response = await call_next(request)
    return response


@app.get("/ping", include_in_schema=False)
async def healthcheck():
    return {"status": "healthy"}


# application endpoint using custom router
app_router = APIRouter(route_class=TimedRoute)


@app_router.post(
    "/",
    description="Detect message toxicity",
    response_model=APIResponse,
    response_description="Toxicity category scores",
    responses={
        500: {"model": ErrorMessage, "description": "Internal server error"},
    },
    dependencies=[
        Depends(CognitoJWTAuthorizer(cognito_pool_id=os.getenv("COGNITO_POOL_ID")))
    ],
)
async def handler(event: APIRequest):
    try:
        return lambda_handler(event.model_dump(exclude_unset=True))
    except ValueError as err:
        error_detail = err.args[0]
        if "TOXICITY_API_ERROR" in error_detail:
            status_code = int(error_detail.replace("TOXICITY_API_ERROR_", ""))
            return JSONResponse(
                status_code=status_code, content={"message": str(error_detail)}
            )
        else:
            raise err
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"message": f"Unhandled server error {traceback.format_exc()}"},
        )


app.include_router(app_router)


# trace uvicorn app https://ddtrace.readthedocs.io/en/stable/integrations.html#asgi
app = TraceMiddleware(app, tracer=instrumentation.tracer)
