import re
import os
import requests
from typing import Any
from langchain import LLMChain, OpenAI, PromptTemplate
import openai 
from bs4 import BeautifulSoup

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from src.models.custom import CustomRetrievalQA

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

class BotChat():
    def __init__(self, temperature: float=0.0, verbose: bool=False, *args, **kwargs) -> None:
        # get chain
        # self.bot_chain = LLMChain(
        #     llm=OpenAI(temperature=temperature, max_tokens=1000), 
        #     verbose=verbose
        
        # )
        
        self.model = "text-davinci-003"
        self.max_tokens = 1000
        self.prompt_template = ""

        
    def __call__(self, context:str,  *args: Any, **kwds: Any) -> Any:
        # TODO: Implement in here (namnh)
        
        self.prompt = self.prompt_template + context
        
        resp = openai.Completion.create(
            model=self.model,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
        )
        
        bot_resp = resp.choices[0].text 
        return bot_resp


class BotSummarize():
    def __init__(self, temperature: float=0.0, verbose: bool=False, *args, **kwargs) -> None:
        
        _template = """{prefix}\n\nCONTEXT:\n\n{context}\n{suffix}"""

        # init prompt
        prompt = PromptTemplate(
            input_variables=["prefix", "context", "suffix"], 
            template=_template
        )

        # get chain
        self.bot_chain = LLMChain(
            llm=OpenAI(temperature=temperature, max_tokens=1000), 
            prompt=prompt,
            verbose=verbose
        )

    def prep_output(self, response: str, output_keys: list):
        # Extract summary
        output = {}
        for output_key in output_keys:
            regex_pattern = r"{}:\s*(.*)\s*\n".format(output_key)
            match = re.search(regex_pattern, response)
            if not match:
                regex_pattern = r"{}:\s*(.*)\s*".format(output_key)
                match = re.search(regex_pattern, response)
            if match:
                output[output_key.lower()] = match.group(1)

        if not output:
            output = {"summary": response}

        return output
    
    def __call__(self, context: str, prompt: str=None, **kwargs):
        # Default prefix and suffix
        _prefix = 'Write a summary with of the following:'
        _suffix = 'Please following format:\n--\n\Format:\nSUMMARY: content summary.'
        
        prefix = _prefix if (not 'prefix' in kwargs) else kwargs['prefix']
        suffix = _suffix if (not 'suffix' in kwargs) else kwargs['suffix']

        # output_keys 
        _output_keys = ["SUMMARY"]
        output_keys = _output_keys if (not 'output_keys' in kwargs) else kwargs['output_keys']

        # Update prompt template
        if prompt:
            # NOTE: Update prompt about prefix
            prefix = prompt + "\n" + prefix

        # prompts  
        response = self.bot_chain.predict(prefix=prefix, context=context, suffix=suffix)

        # preprocessing output
        output = self.prep_output(response=response, output_keys=output_keys)

        return output


class BotQuiz():
    pass


class BotSLearn():
    def __init__(self) -> None:
        self.chatbot = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0, model_name="gpt-3.5-turbo", max_tokens=1000
            ), 
            chain_type="stuff", 
            retriever=FAISS.load_local("resources/faiss/slearn_pdf", OpenAIEmbeddings())
                .as_retriever(search_type="similarity", search_kwargs={"k":1})
        )

        template = """Bạn là một chatbot hỗ trợ trả lời các thông tin về dự án SLearn.
        
        ""SLearn là dự được thành lập bởi AIClub@UIT Team gồm 4 thành viên: Nguyễn Hoài Nam, Nguyễn Minh Lý, Nguyễn Khắc Thái, Trần Thị Thanh Hiếu.""
        ""AIClub@UIT là viết tắt của Câu lạc bộ Trí tuệ nhân tạo - Khoa Khoa học Máy tính - Trường Đại học Công nghệ Thông tin - Đại học Quốc gia TP. Hồ Chí Minh""
        
        -------------
        
        {query}"""
        
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

    def __call__(self, query: str):
        return self.chatbot.run(self.prompt.format(query=query))

class BotDefault():
    def __init__(self) -> None:
        self.turbo_llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo'
        )

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="{query}",
        )
        
        self.bot_chain = LLMChain(
            prompt=self.prompt,
            llm=self.turbo_llm, 
            verbose=True
        )
            
    def __call__(self, query: str) -> Any:

        return self.bot_chain.run(self.prompt.format(query=query))

class BotRealtime():
    def __init__(self) -> None:
        page_getter = WebPageTool()

        search = GoogleSearchAPIWrapper()
        prompt_tool_search = "useful for when you need to answer questions about current events. You should ask targeted questions"
        search_tool = Tool(
                name = "Google Search",
                description=prompt_tool_search,
                func=search.run
            )     
        
        tools = [search_tool, page_getter]   
        llm = ChatOpenAI(temperature=0)
        self.conversational_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
    def __call__(self, query) -> Any:
        return self.conversational_agent.run(query)

class BotOraichain():
    def __init__(self) -> None:
        self.gg_search = GoogleSearchAPIWrapper()
        llm = ChatOpenAI(temperature=0)
        
        prompt_tool_search = "useful for when you need to answer questions about oraichain, oraichain platform, crytyco, blockchain, nft, ..."
        search_tool = Tool(
                name = "Google Search",
                description=prompt_tool_search,
                func=self.gg_search.run
            )

        tools = [search_tool]
        self.agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def get_cite(self, query):
        resp = self.gg_search.results(query=query, num_results=3) 
        return resp 
    
    def __call__(self, query) -> Any:
        ans = self.agent.run(query)
        cite = self.get_cite(query=query)
        
        return ans, cite
    

class BotCourseBlockchain():
    def __init__(self) -> None:
        self.chatbot = CustomRetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0, model_name="gpt-3.5-turbo", max_tokens=1000
            ), 
            chain_type="stuff", 
            retriever=FAISS.load_local("resources/faiss/blockchain_anyw", OpenAIEmbeddings())
                .as_retriever(search_type="similarity", search_kwargs={"k":3})
        )

        template = """{query}"""
        
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )
        

    def __call__(self, query: str):
        ans = self.chatbot.run(self.prompt.format(query=query))  
        doc = self.chatbot.action_logs[0]
        try:
            cite = dict(
                title = doc.metadata['title'],
                link = doc.metadata['link'],
                snippet = doc.page_content,
            )
        except:
            cite = []
        return ans, cite  
    

class BotCheckComment():
    def __init__(self) -> None:
        self.turbo_llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo'
        )

        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a system for checking negative comments, where users input a comment and you have the task of analyzing and evaluating whether the comment is negative or not.

--
TRUE: if it contains negative, offensive, provocative, derogatory, suggestive content, etc.

FALSE: if it is a normal response to a question.

---
Please remember that you only need to ANSWER WITH SINGLE WORD, ((either TRUE or FALSE)), keep this in mind.

Here is the user's comment:

{query}""",
        )
        
        self.bot_chain = LLMChain(
            prompt=self.prompt,
            llm=self.turbo_llm, 
            verbose=True
        )
            
    def __call__(self, query: str) -> Any:

        ans = self.bot_chain.run(self.prompt.format(query=query))
        
        if ans.lower() == 'true':
            return dict(negative=True)
        else:
            return dict(negative=False)
    