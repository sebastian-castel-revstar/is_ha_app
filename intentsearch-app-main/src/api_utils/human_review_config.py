import os


class BaseConfig:
    ENV = os.environ["env"]
    JOB_NAME_PREFIX = f"h-{ENV}-ds-genai-intent-search-labeling-job"
    BUCKET_NAME = f"h-{ENV}-genai-events"
    ARTIFACT_PREFIX = "intent-search-human-review-artifacts"
    OUTPUT_PREFIX = "intent-search-human-review-output"

    WORKER_PER_OBJECT = 1
    TASK_TIME_LIMIT_IN_SECONDS = 28800
    TASK_AVAILABILITY_LIFETIME_IN_SECONDS = 86400
    MAX_CONCURRENT_TASKS = 5

    WORKTEAM_ARN = os.environ["workteam_arn"]
    EXECUTION_ROLE_ARN = os.environ["execution_role_arn"]

    @classmethod
    def get_config(cls):
        return {
            "worker_per_object": cls.WORKER_PER_OBJECT,
            "task_time_limit_in_seconds": cls.TASK_TIME_LIMIT_IN_SECONDS,
            "task_availability_lifetime_in_seconds": cls.TASK_AVAILABILITY_LIFETIME_IN_SECONDS,
            "max_concurrent_tasks": cls.MAX_CONCURRENT_TASKS,
            "workteam_arn": cls.WORKTEAM_ARN,
            "execution_role_arn": cls.EXECUTION_ROLE_ARN,
        }


class NLConfig(BaseConfig):
    API_NAME = "nlapi"

    @classmethod
    def get_config(cls):
        base_config = super().get_config()
        return {
            **base_config,
            # SHARED CONFIG
            "labeling_job_name": f"{BaseConfig.JOB_NAME_PREFIX}-{cls.API_NAME}",
            "labeling_output_location": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.OUTPUT_PREFIX}/{cls.API_NAME}",
            "labels_s3_uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-human-review-instructions.json",
            # API SPECIFIC
            "task_title": "Intent_Search_NL_API",
            "task_description": "Natural Language API Human Labeling Job",
            "task_keywords": [
                "Intent Search",
                "Named Entity Recognition",
                "NER",
                "Natural Language",
            ],
            "input_config_data_source": {
                "SnsDataSource": {
                    "SnsTopicArn": os.environ[f"{cls.API_NAME}_sns_topic_arn"]
                },
                "S3DataSource": {
                    "ManifestS3Uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-input-manifest.json"
                },
            },
            "ui_config": {"HumanTaskUiArn": os.environ["ui_template_arn"]},
            "pre_human_task_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:PRE-NamedEntityRecognition",
            "annotation_consolidation_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:ACS-NamedEntityRecognition",
        }


class QCConfig(BaseConfig):
    API_NAME = "qcapi"

    @classmethod
    def get_config(cls):
        base_config = super().get_config()
        return {
            **base_config,
            # SHARED CONFIG
            "labeling_job_name": f"{BaseConfig.JOB_NAME_PREFIX}-{cls.API_NAME}",
            "labeling_output_location": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.OUTPUT_PREFIX}/{cls.API_NAME}",
            "labels_s3_uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-human-review-instructions.json",
            # API SPECIFIC
            "task_title": "Intent_Search_QC_API",
            "task_description": "Query Classifier API Human Labeling Job",
            "task_keywords": ["Intent Search", "Query Classifier"],
            "input_config_data_source": {
                "SnsDataSource": {
                    "SnsTopicArn": os.environ[f"{cls.API_NAME}_sns_topic_arn"]
                }
            },
            "ui_config": {
                "UiTemplateS3Uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-tagging-instructions.html"
            },
            "pre_human_task_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClass",
            "annotation_consolidation_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClass",
        }


class ToxConfig(BaseConfig):
    API_NAME = "toxapi"

    @classmethod
    def get_config(cls):
        base_config = super().get_config()
        return {
            **base_config,
            # SHARED CONFIG
            "labeling_job_name": f"{BaseConfig.JOB_NAME_PREFIX}-{cls.API_NAME}",
            "labeling_output_location": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.OUTPUT_PREFIX}/{cls.API_NAME}",
            "labels_s3_uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-human-review-instructions.json",
            # API SPECIFIC
            "task_title": "Intent_Search_Toxicity_API",
            "task_description": "Toxicity API Human Labeling Job",
            "task_keywords": ["Intent Search", "Toxicity"],
            "input_config_data_source": {
                "SnsDataSource": {
                    "SnsTopicArn": os.environ[f"{cls.API_NAME}_sns_topic_arn"]
                }
            },
            "ui_config": {
                "UiTemplateS3Uri": f"s3://{BaseConfig.BUCKET_NAME}/{BaseConfig.ARTIFACT_PREFIX}/{cls.API_NAME}-tagging-instructions.html"
            },
            "pre_human_task_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClassMultiLabel",
            "annotation_consolidation_lambda_arn": "arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClassMultiLabel",
        }
