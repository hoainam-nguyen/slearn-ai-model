from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain.llms import OpenAI
from langchain.agents import tool

from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import BaseTool

llm = OpenAI(temperature=0.0)

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


################### Tool Search
search = GoogleSearchAPIWrapper()

prompt_tool_search = "useful for when you need to answer questions about current events. You should ask targeted questions"
search_tool = Tool(
        name = "Google Search",
        description=prompt_tool_search,
        func=search.run
    )


#################################
botqa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=1000
    ), 
    chain_type="stuff", 
    retriever=FAISS.load_local("resources/faiss/slearn_pdf", OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)

#################################
prompt_tool_bot = "useful when ask about SLearn"
bot_tool = Tool(
        name = "Knowlegde Search",
        description=prompt_tool_bot,
        func=botqa.run
    )


tools = [search_tool, bot_tool]

# CALL
query = "Dự án SLearn là gì ?"


# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# print(agent.run(query))
# a = agent(query)
# print(a)


#############################
from bs4 import BeautifulSoup
import requests

def stripped_webpage(webpage):
    response = requests.get(webpage)
    html_content = response.text

    def strip_html_tags(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        stripped_text = soup.get_text()
        stripped_text = stripped_text.split()
        stripped_text = ' '.join(stripped_text)
        return stripped_text

    stripped_content = strip_html_tags(html_content)

    if len(stripped_content) > 4000:
        stripped_content = stripped_content[:4000]
    return stripped_content

class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content
    
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

# page_getter = WebPageTool()

# tools = [page_getter]

# conversational_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# result = conversational_agent.run("Tóm tắt nội dung trên https://blockai.care/")
# print(result)

###########################

search_a = GoogleSearchAPIWrapper()
res = search_a.results("What is token Oraichain support ?", num_results=3)
print(res)

##############################
search = GoogleSearchAPIWrapper()

prompt_tool_search = "useful for when you need to answer questions about current events, oraichain, oraichain platform, crytyco, blockchain, nft, ..."
search_tool = Tool(
        name = "Google Search",
        description=prompt_tool_search,
        func=search.run
    )

tools = [search_tool]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

res = agent.run("What is token Oraichain support ?")
print(res)
pass