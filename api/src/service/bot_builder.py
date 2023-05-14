import sys 
sys.path.append('/home/mmlab/Projects/dev/slearn-oraichain/api')

import os
import re
import asyncio

from copy import deepcopy
from langchain import LLMMathChain, OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain import FAISS
from langchain.agents import Tool, ConversationalAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.sql_database import SQLDatabase

from langchain.agents import initialize_agent, AgentType

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from src.service.bot import CustomBaseLLMChain, CustomBaseVectorDBQA, CustomConversationalAgent
from src.service.customGGSearchTool import CustomGoogleSearchAPIWrapper

from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import AgentExecutor, ConversationalAgent
class BotBuilder:
    def __init__(self, *args, **kwargs) -> None:
        self.support_tools = None
        self.action_logs = []
        self.agent = None
        self.suffix = None
        self.prefix = None
        self.agent_executor = None,
        self.default_completion_llm = OpenAI(temperature=0, max_tokens=-1) 
        self.default_chat_llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo', max_tokens=None) 
        self.bot_llm = {}
        self.gg_tool = CustomGoogleSearchAPIWrapper()
        

    def get(self, bot_id):
        return self.bot_llm[bot_id]

    def _create_tool(self, tools_config: dict, llm_config: dict):
        name = tools_config['name']
        description = tools_config['description']
        path: str = f"{os.path.abspath(os.getcwd())}/resource/faiss"
        tool_type = tools_config.get('type')

        run_func = None
        arun_func = None
        gg_tool = None

        if tool_type == 'lamda':
            run_func = lambda _: tools_config['value']
        elif tool_type == 'gg_search':
            gg_tool = self.gg_tool            
        elif tool_type == 'faiss_search':
            try:
                index_config = tools_config['index_config']

                run_func = CustomBaseVectorDBQA.from_chain_type(
                    llm=OpenAI(
                        temperature=llm_config['temperature'],
                        max_tokens=llm_config['max_tokens'],
                        max_retries=12
                    ),
                    chain_type=index_config["chain_type"],
                    vectorstore=FAISS.load_local(
                        folder_path=index_config["index_path"],
                        embeddings=OpenAIEmbeddings()
                    ),
                    k=index_config["faiss_knn"],
                    return_source_documents=index_config["return_source_documents"]
                )
            except Exception as e:
                return None

        else:  # invalid tool or none
            return None

        # return Tool(name=name, func=run_func, description=description), tool_type
        return dict(
            tool_type = tool_type,
            tool_func = Tool(name=name, func=run_func, description=description, coroutine = arun_func),
            gg_tool=gg_tool
        )


    def create_bot(self, input_dict: dict, bot_id):
        """prompt_guidance"""
        """
            input_dict = {
                "prefix": 
                "suffix": 
                "tools": [{
                    "name":"Knowledge Search",
                    "slug": "faiss_search",
                    "description":"prioritize this tool, useful when user want to ask about crypto, anything related with Paul Veradittakit and his blogs"
                }],
                "LLMChain_config": {
                    "temperature":
                    "max_tokens":
                }
                "verbose"

            }
        """
        prefix = input_dict["prefix"]
        suffix = input_dict["suffix"]

        tool_maps = {}

        tools = []
        for tool in input_dict['tools']:
            tool_dict = self._create_tool(tool, input_dict["LLMChain_config"])
            if not tool_dict:
                continue
            _tool = tool_dict['tool_func']


            if tool_dict['tool_type'] in ['faiss_search']:
                key = f'{tool_dict["tool_type"]}_' + _tool.name.lower().replace(' ', '_') 
                tool_maps[key] = _tool.func
                # _tool.coroutine = _tool.func.arun # NOTE: use async 
                # _tool.func = _tool.func.run
            elif tool_dict['tool_type'] == 'gg_search':
                key = f'{tool_dict["tool_type"]}_' + _tool.name.lower().replace(' ', '_') 
                tool_maps[key] = tool_dict['gg_tool']

            tools.append(_tool)

        tool_names = [tool.name for tool in tools]

        # Get input variables
        _prompt_template = prefix + suffix
        input_variables = re.findall(r"{([^}]*)}", _prompt_template)
        
        prompt = ConversationalAgent.create_prompt(
            tools=tools,
            prefix=prefix,  # inject personalities
            suffix=suffix,  # inject guidance answer
            input_variables=input_variables
        )


        if input_dict["LLMChain_config"] is not None:
            chat_llm = CustomBaseLLMChain(
                llm=ChatOpenAI(
                    temperature=input_dict["LLMChain_config"]["temperature"],
                    max_tokens= input_dict["LLMChain_config"]["max_tokens"] \
                                    if input_dict["LLMChain_config"]["max_tokens"] > 0 \
                                    else None,  
                    model_name='gpt-3.5-turbo'
                ),
                prompt=prompt
            )
        else:
            chat_llm = CustomBaseLLMChain(
                llm=self.default_chat_llm, 
                prompt=prompt
            )

        # agent = CustomConversationalAgent(llm_chain=chat_llm, allowed_tools=tool_names)
        # agent = ConversationalAgent(llm_chain=chat_llm, allowed_tools=tool_names)
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
                
        agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory,
        )
     

        self.bot_llm[bot_id] = dict(agent=agent, tools=tools, tools_map=tool_maps, tool_names=tool_names, config=input_dict)
    

if __name__=="__main__":
    BOT_MAP = BotBuilder()
    
    input_dict = dict(
        prefix = "{input}\n{agent_scratchpad}",
        suffix = "",
        tools = [
            dict(
                name = "Knowledge Search",
                type = "faiss_search",
                description = "prioritize this tool, useful when user want to ask about crypto, anything related with Paul Veradittakit and his blogs",
                index_config = dict(
                    chain_type="stuff",
                    index_path="resources/faiss/VV_html",
                    return_source_documents=False,
                    faiss_knn=1
                )
            )
        ],
        LLMChain_config = dict(
            temperature = 0.0,
            max_tokens = 1000
        ),
        verbose=True
    )

    BOT_MAP.create_bot(input_dict=input_dict, bot_id='test')
    bot = BOT_MAP.get('test')
    res = bot['agent'].run(input='How many price blockchian to day', output_keys=['output'])
    print(res)
    pass



