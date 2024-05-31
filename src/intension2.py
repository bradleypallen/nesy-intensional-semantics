import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.output_parsers import RegexParser
from langchain_core.prompts import PromptTemplate

class Intension:
    """Represents a zero-shot chain-of-thought implementing an intension for predicates."""

    PROMPT_TEMPLATE = """
Is the following logical statement true or false? {predicate}({arguments})
Evaluate the truth value of this statement in a hypothetical world where the following is true:
{world}
Let's think step by step. Provide a rationale for your decision, then based on that rationale,
provide an answer of 1 if true, otherwise provide an answer of 0.
###
Rationale: {{rationale}}
Answer: {{answer}}
"""

    PROMPT = PromptTemplate(input_variables=["predicate", "arguments", "world"], template=PROMPT_TEMPLATE)

    OUTPUT_PARSER = RegexParser(
        regex=r"(?is).*Rationale:\**\s*(.*?)\n+.*Answer:\**\s*(0|1)",
        output_keys=["rationale", "answer"],
        default_output_key="rationale"
    )
    
    def __init__(self, model, temperature=0.1, timeout=120, endpoint=None):
        """
        Initializes a intension for a concept.
        
        Parameters:
            model: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
            timeout: Timeout for the API call in seconds (default 120)).
            endpoint: A URL for a dedicated inference endpoint for the model (default None)
         """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.endpoint = endpoint
        if endpoint:
            self.llm = HuggingFaceEndpoint(
                endpoint_url=endpoint,
                temperature=temperature, 
                timeout=timeout,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        else:
            self.llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=temperature, 
                timeout=timeout,
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
        self.chain = LLMChain(llm=self.llm, prompt=self.PROMPT, output_parser=self.OUTPUT_PARSER)

