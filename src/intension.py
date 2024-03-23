import os, sys
from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint

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
    
    def __init__(self, model_name="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes a classification procedure for a concept, given a unique identifier, a term, and a definition.
        
        Parameters:
            model_name: The name of the model to be used for zero shot CoT classification (default "gpt-4").
            temperature: The temperature parameter for the model (default 0.1).
         """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._llm(model_name, temperature)
        self.chain = LLMChain(llm=self.llm, prompt=self.PROMPT, output_parser=self.OUTPUT_PARSER)

    def _llm(self, model_name, temperature):
        if model_name in [ "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview" ]:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name in [ "claude-3-opus-20240229" ]:
            return ChatAnthropic(
                temperature=temperature, 
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], 
                model_name=model_name
            )
        elif model_name in [ "gemini-1.0-pro" ]:
            return ChatGoogleGenerativeAI(
                temperature=temperature, 
                google_api_key=os.environ["GOOGLE_API_KEY"], 
                model=model_name
            )
        elif model_name in [
            "meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1", 
            "mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-7b-it", "google/gemma-2b-it" ]:
            llm = None
            stdout = sys.stdout
            try:
                with open(os.devnull, 'w') as devnull:
                    sys.stdout = devnull
                    llm = HuggingFaceEndpoint(
                        repo_id=model_name, 
                        temperature=temperature, 
                        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
                    )
            except Exception as e:
                print(f"An error occurred: {str(e)}")
            finally:
                sys.stdout = stdout
            return llm
        else:
            raise Exception(f'Model {model_name} not supported')
