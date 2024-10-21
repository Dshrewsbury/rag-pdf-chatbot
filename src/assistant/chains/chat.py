from typing import Any
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from llama_cpp import Llama

from .._assistant import Assistant

class Chat(Assistant):
    """
    Chat data structure.
    """
    def __init__(self, model_path: str):
        # Refactor hard-coded values into config file
        # Also context window being so large is making responses very slow
        self.model = LlamaCpp(
            verbose=True,
            model_path=model_path, 
            callbacks=[self.handler],
            n_batch=256,
            n_ctx=2048,
            top_k=100,
            top_p=0.37,
            temperature=0.8,
            max_tokens=200,
        )

    def new_chain(self, **kwargs: Any):
        human_prefix=kwargs.get("human_prefix", "Human")

        # ConversationBufferWindowMemory acting as short-term memory
        return ConversationChain(
            llm=self.model,
            prompt=self.get_prompt_template(human_prefix),
            callbacks=[self.handler],
            memory=ConversationBufferWindowMemory(
                        k=1,
                        human_prefix=human_prefix
                    )
        )

    def get_prompt_template(self, human_prefix: str = "User"):
        # Instruction section to guide the assistant
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Add clear sections for input (user question) and context
        instruction = f"Recent Chat History:\n\n{{history}} \n\n {{input}}"
        system_prompt = B_SYS + """
        You are a helpful assistant on the Llama 2 paper. Answer the user's questions related to the paper using the provided context on the paper if relevant.
        Always separate the context from the question and focus on answering the question directly.
        Do not mention that you were provided context just answer the question. Do not ask for more information unless it is absolutely necessary.
        """ + E_SYS

        # Combine everything into the final prompt template
        template = B_INST + system_prompt + instruction + E_INST + "\nAI:"

        return PromptTemplate(
            template=template,
            input_variables=["history", "input"],
            partial_variables={"human_prefix": human_prefix}
        )
