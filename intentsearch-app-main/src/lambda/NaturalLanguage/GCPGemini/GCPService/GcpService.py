import json
import os
import tempfile
import time
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from GCPGemini.GCPService.Abstract import SummaryService
from GCPGemini.GCPService.GcpConfig import GcpConfig

"""
GCP Embeddings is a service that provides pre-trained embeddings for text data.
This class provides a simple interface to the GCP Embeddings service.
"""


class GcpSummaryService(SummaryService):
    def __init__(self, config: GcpConfig):
        """
        Initialize the GCP summary object.
        Args:
            api_key (Optional[str]): API key to access GCP API.
            service_account_info (Optional[Dict]): Service Account Info to access GCP API.
            model_id (str): The model ID to use.
            logger (Optional[Logger]): logger being used.
        """
        # Initialize logger
        self.logger = config.logger
        self.model_id = config.model_id
        self.service_account_info = config.service_account_info
        self.cached_gcp_cred = config.cached_gcp_cred

        try:
            self.ensure_gcp_credentials()
            self.client = genai.Client(
                vertexai=True, project=config.project_id, location=config.location
            )
        except Exception as e:
            self.logger.error("Failed to configure credentials: %s", e)
            raise

    def ensure_gcp_credentials(
        self, cache_path: str = "/tmp/gcp_sa.json", cache_ttl: int = 12 * 3600
    ) -> str:
        """
        Fetch GCP credentials from AWS Secrets Manager or use cached file.
        Always sets GOOGLE_APPLICATION_CREDENTIALS and returns the path.
        """
        write_required = True

        if self.cached_gcp_cred:
            if self.cached_gcp_cred == self.service_account_info:
                self.logger.info("Using cached GCP credentials at %s", cache_path)
                write_required = False
            else:
                self.logger.info("Cached credentials differ — updating %s", cache_path)

        if write_required:
            self.cached_gcp_cred = self.service_account_info
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=os.path.dirname(cache_path)
            ) as tmp:
                json.dump(self.service_account_info, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp.name, cache_path)
            self.logger.info("Cached GCP credentials to %s", cache_path)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cache_path
        return cache_path

    def get_response(
        self,
        prompt_instructions: str,
        max_output_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.999,
        top_k: int = 40,
        stopSequences: Optional[List[str]] = None,
        candidateCount: int = 1,
        presencePenalty: float = 0,
        frequencyPenalty: float = 0,
        echo: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Retrieves summary for the provided prompt instructions.
        Args:
            prompt_instructions (str): Instructions and text to summarize.
            max_output_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_p (float): Top p value for nucleus sampling.
            top_k (int): Top k value for sampling.
            stopSequences (Optional[List[str]]): List of stop sequences.
            candidateCount (int): Number of candidates to generate.
            presencePenalty (float): Presence penalty.
            frequencyPenalty (float): Frequency penalty.
            echo (bool): Echo the prompt.
            seed (Optional[int]): Seed for random number generator.
        Return:
            Dict: Summary response.
        """
        return self(
            prompt_instructions,
            max_output_tokens,
            temperature,
            top_p,
            top_k,
            stopSequences,
            candidateCount,
            presencePenalty,
            frequencyPenalty,
            echo,
            seed,
        )

    def __call__(
        self,
        prompt_instructions: str,
        max_output_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.999,
        top_k: int = 40,
        stopSequences: Optional[List[str]] = None,
        candidateCount: int = 1,
        presencePenalty: float = 0,
        frequencyPenalty: float = 0,
        echo: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Returns Text Summary
        Args:
            prompt_instructions (str): Text to summarize.
            max_output_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_p (float): Top p value for nucleus sampling.
            top_k (int): Top k value for sampling.
            stopSequences (Optional[List[str]]): List of stop sequences.
            candidateCount (int): Number of candidates to generate.
            presencePenalty (float): Presence penalty.
            frequencyPenalty (float): Frequency penalty.
            echo (bool): Echo the prompt.
            seed (Optional[int]): Seed for random number generator.
        Return:
            Dict: Text summary.
        """
        try:
            # Record start time
            start_time = time.time()
            self.logger.info("Starting model invocation")

            contents = [
                types.Content(
                    role="user", parts=[types.Part.from_text(text=prompt_instructions)]
                )
            ]
            cfg = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                labels={"Cost_Center": "intent search"},
            )

            response = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_id, contents=contents, config=cfg
            ):
                response += chunk.text

            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info("Model invocation duration: %s seconds", duration)

            return response

        except Exception as e:
            self.logger.error("Failed to get summary: %s", e)
            raise
