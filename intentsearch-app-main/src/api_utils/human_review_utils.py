import json
import os
from datetime import datetime

import requests
from aws_lambda_powertools import Logger, Tracer

logger = Logger(log_uncaught_exceptions=True)
tracer = Tracer()


@tracer.capture_method
def get_lambda_tags(lambda_client, lambda_arn: str) -> list:
    tags_response = lambda_client.list_tags(Resource=lambda_arn)
    return [
        {"Key": k, "Value": v}
        for k, v in tags_response.get("Tags", {}).items()
        if not k.lower().startswith(("aws:", "lambda:"))
    ]


def get_ecs_tags(ecs_client, resource_arn: str) -> list:
    task_arn = requests.get(
        os.getenv("ECS_CONTAINER_METADATA_URI_V4") + "/taskWithTags"
    ).json()["TaskARN"]
    tags_response = ecs_client.list_tags_for_resource(resourceArn=task_arn)
    return [{"key": k, "value": v} for k, v in tags_response.get("tags", {}).items()]


@tracer.capture_method
def create_labeling_job(
    sagemaker,
    tags: list,
    labeling_job_name: str,
    labeling_output_location: str,
    labels_s3_uri: str,
    input_config_data_source: dict,
    ui_config: dict,
    workteam_arn: str,
    execution_role_arn: str,
    pre_human_task_lambda_arn: str,
    annotation_consolidation_lambda_arn: str,
    task_keywords: list,
    task_title: str,
    task_description: str,
    worker_per_object: int,
    task_time_limit_in_seconds: int,
    task_availability_lifetime_in_seconds: int,
    max_concurrent_tasks: int,
) -> None:
    try:
        timestamp_str = datetime.now().strftime("%y%m%d%H%M")
        labeling_job_name = f"{labeling_job_name}-{timestamp_str}"
        sagemaker.create_labeling_job(
            LabelingJobName=labeling_job_name,
            LabelAttributeName="label",
            InputConfig={
                "DataSource": input_config_data_source,
                "DataAttributes": {
                    "ContentClassifiers": [
                        "FreeOfPersonallyIdentifiableInformation",
                        "FreeOfAdultContent",
                    ]
                },
            },
            OutputConfig={"S3OutputPath": labeling_output_location},
            RoleArn=execution_role_arn,
            LabelCategoryConfigS3Uri=labels_s3_uri,
            HumanTaskConfig={
                "WorkteamArn": workteam_arn,
                "UiConfig": ui_config,
                "PreHumanTaskLambdaArn": pre_human_task_lambda_arn,
                "TaskKeywords": task_keywords,
                "TaskTitle": f"{task_title}_{timestamp_str}",
                "TaskDescription": task_description,
                "NumberOfHumanWorkersPerDataObject": worker_per_object,
                "TaskTimeLimitInSeconds": task_time_limit_in_seconds,
                "TaskAvailabilityLifetimeInSeconds": task_availability_lifetime_in_seconds,
                "MaxConcurrentTaskCount": max_concurrent_tasks,
                "AnnotationConsolidationConfig": {
                    "AnnotationConsolidationLambdaArn": annotation_consolidation_lambda_arn
                },
            },
            Tags=tags,
        )
    except Exception as e:
        logger.exception(
            "An unexpected error occurred when creating the ground truth labeling job: %s",
            e,
        )
        raise ValueError("CREATE_LABELING_JOB_ERROR") from e


@tracer.capture_method
def post_to_labeling_job(
    sagemaker,
    sns,
    query: str,
    labeling_job_configuration: dict,
    sns_topic_arn: str,
    tags: list,
) -> None:
    try:
        response = sagemaker.list_labeling_jobs(
            NameContains=labeling_job_configuration["labeling_job_name"]
        )
        labeling_jobs_summary_list = response.get("LabelingJobSummaryList", [])
        logger.info(labeling_jobs_summary_list)

        filtered_labeling_jobs_summary_list = [
            labeling_job
            for labeling_job in labeling_jobs_summary_list
            if labeling_job["LabelingJobStatus"] in ["Initializing", "InProgress"]
        ]

        if not filtered_labeling_jobs_summary_list:
            logger.info(
                "Creating Labeling Job for %s",
                labeling_job_configuration["labeling_job_name"],
            )
            create_labeling_job(
                **labeling_job_configuration, tags=tags, sagemaker=sagemaker
            )

        formatted_message = {"source": query}
        sns.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps(formatted_message),
        )
        logger.info(
            "Published message for labeling job %s",
            labeling_job_configuration["labeling_job_name"],
        )
    except Exception as e:
        logger.exception(
            "An unexpected error occurred when posting tasks to the ground truth labeling job: %s",
            e,
        )
