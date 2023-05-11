# !pip install -q -U wandb langchain openai wikipedia
from dotenv import load_dotenv
import tiktoken
import openai
import os


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain import Wikipedia
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

docstore = DocstoreExplorer(Wikipedia())

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup"
    )
]

llm = ChatOpenAI(
    temperature=0, 
    model_name="gpt-3.5-turbo",
    openai_api_key=openai.api_key
)
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

question = "Who directed the movie about Alexander Supertramp?"

from wandb.integration.langchain import WandbTracer

# again, you may get an error, but the tracing will still work!
wandb_config = {"project": "wandb_prompts_react_demo"}
react.run(question, callbacks=[WandbTracer(wandb_config)])
