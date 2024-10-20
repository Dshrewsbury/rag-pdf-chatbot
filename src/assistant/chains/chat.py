from typing import Any
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
#from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from llama_cpp import Llama

from .._assistant import Assistant

class Chat(Assistant):
    """
    Chat data structure.
    """
    def __init__(self, model_path: str):
        self.model = LlamaCpp(
            verbose=True,
            model_path=model_path, 
            callbacks=[self.handler],
            n_gpu_layers=25,
            n_batch=256,
            n_ctx=1024,
            top_k=100,
            top_p=0.37,
            temperature=0.7,
            max_tokens=200,
        )

    def new_chain(self, **kwargs: Any):
        human_prefix=kwargs.get("human_prefix", "Human")
        user_input = "what is love"  # This is where you store the user input
        #context ="Brand Engagement Network is an AI Company that everyone loves"  # This should contain your retrieved context

        return ConversationChain(
            llm=self.model,
            prompt=self.get_prompt_template(human_prefix),
            callbacks=[self.handler],
            memory=ConversationBufferWindowMemory(
                k=3,
                human_prefix=human_prefix
            )
        )

    # def get_prompt_template(self, human_prefix: str = "User"):
    #
    #     # Include the retrieved documents in the prompt template
    #     B_INST, E_INST = "[INST]", "[/INST]"
    #     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    #
    #     instruction = f"Recent Chat History:\n\n{{history}} \n {{human_prefix}}: {{input}}"
    #     system_prompt = B_SYS +"You are a helpful assistant, you always only answer for the assistant then you stop. Answer the user's question labeled by USER QUESTION, using the CONTEXT, and also the LONGTERM memory and history if it is relevant to the user's question. Do not ask for more information"+ E_SYS
    #
    #     template =  B_INST + system_prompt + instruction + E_INST + "\nAI:"
    #
    #     return PromptTemplate(
    #         template=template,
    #         input_variables=["history", "input"],
    #         partial_variables={"human_prefix": human_prefix}
    #     )

    def get_prompt_template(self, human_prefix: str = "User"):
        # Instruction section to guide the assistant
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Add clear sections for input (user question) and context
        instruction = f"Recent Chat History:\n\n{{history}} \n\nContext (use this to help answer the question):\n {{input}}"
        system_prompt = B_SYS + """
        You are a helpful assistant. Answer the user's question using the provided context if relevant.
        Always separate the context from the question and focus on answering the question directly.
        Do not combine the user's input with the context. Do not ask for more information unless it is absolutely necessary.
        """ + E_SYS

        # Combine everything into the final prompt template
        template = B_INST + system_prompt + instruction + E_INST + "\nAI:"

        return PromptTemplate(
            template=template,
            input_variables=["history", "input"],
            partial_variables={"human_prefix": human_prefix}
        )
