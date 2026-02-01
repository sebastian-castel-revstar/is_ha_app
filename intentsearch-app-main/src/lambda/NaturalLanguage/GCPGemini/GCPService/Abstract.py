# Author: Sean Iannuzzi
# Created: August 6, 2024

from abc import ABC, abstractmethod
from typing import List, Optional

"""
SummaryService is an abstract class that provides an interface for summary services.
"""


class SummaryService(ABC):
    @abstractmethod
    def get_response(
        self,
        prompt_instructions: str,
        max_tokens: int = 3072,
        stop_sequences: List[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.999,
        top_k: int = 0,
        candidate_count: int = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        echo: bool = False,
        seed: Optional[int] = None,
        bedrock_trace: str = "DISABLED",
        guardrail_identifier: Optional[str] = None,
        guardrail_version: Optional[str] = None,
    ) -> str:
        """
        Abstract method to get a summary for the provided text.

        Args:
            prompt_instructions (str): Instructions and text to summarize.
            max_tokens (int): Maximum number of tokens to generate.
            stop_sequences (List[str]): List of stop sequences.
            temperature (float): Temperature for sampling.
            top_p (float): Top-p value for sampling.
            top_k (int): Top-k value for sampling.
            candidate_count (int): Number of candidates to generate.
            presence_penalty (float): Presence penalty.
            frequency_penalty (float): Frequency penalty.
            echo (bool): Echo the prompt.
            seed (Optional[int]): Seed for random number generator.
            bedrock_trace (str): Trace level for Bedrock.
            guardrail_identifier (Optional[str]): Identifier for guardrail.
            guardrail_version (Optional[str]): Version of the guardrail.

        Returns:
            str: Text summary.
        """
        pass
