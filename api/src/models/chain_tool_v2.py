from dotenv import load_dotenv
load_dotenv()

import sys 
sys.path.append('/home/mmlab/Projects/dev/slearn-oraichain/api')
import os
import requests

from langchain.llms import OpenAI
from langchain.agents import tool

from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

llm = OpenAI(temperature=0.0)

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from src.service.bot import CustomBaseLLMChain, CustomBaseVectorDBQA, CustomConversationalAgent

################### Tool Search
search = GoogleSearchAPIWrapper()

prompt_tool_search = "useful for when you need to answer questions about current events. You should ask targeted questions"
search_tool = Tool(
        name = "Google Search",
        description=prompt_tool_search,
        func=search.run
    )


#################################
botqa = CustomBaseVectorDBQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type='stuff',
    vectorstore=FAISS.load_local("resources/faiss/VV_html", embeddings=OpenAIEmbeddings()),
    k=3,
    return_source_documents=True
)

#################################
prompt_tool_bot = "useful when ask about blockchain, crypto, nft"
bot_tool = Tool(
        name = "Knowlegde Search",
        description=prompt_tool_bot,
        func=botqa.run
    )


tools = [search_tool, bot_tool]

# CALL
query = "What is NFT?"
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.run(query))

# print(botqa.run('what is NFT'))

pass
