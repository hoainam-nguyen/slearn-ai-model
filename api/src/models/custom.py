import os, dotenv
import inspect
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.manager import Callbacks, CallbackManager

dotenv.load_dotenv()
#######################
class CustomChain(Chain):
    action_logs = []
    def run(self, *args: Any, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        # if len(self.output_keys) != 1:
        #     raise ValueError(
        #         f"`run` not supported when there is not exactly "
        #         f"one output key. Got {self.output_keys}."
        #     )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return self(args[0], callbacks=callbacks)[self.output_keys[0]]

        if kwargs and not args:
            return self(kwargs, callbacks=callbacks)[self.output_keys[0]]

        if not kwargs and not args:
            raise ValueError(
                "`run` supported with either positional arguments or keyword arguments,"
                " but none were provided."
            )

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )
    
    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
    ) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            inputs,
        )
        try:
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
            self.action_logs.append(outputs)
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)
        return self.prep_outputs(inputs, outputs, return_only_outputs)


#######################

class CustomRetrievalQA(RetrievalQA, CustomChain):
    pass 