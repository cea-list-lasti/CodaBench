from typings import InferenceConfig   
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

class Generation(ABC):
    @abstractmethod
    def generate(self, text : str, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, config : InferenceConfig) -> str:
        pass


class HFBaseGeneration(Generation):
    def generate(self, text : str, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, config : InferenceConfig) -> str:
        """ Generate a batch for huggingface standard models

            Args:
                text (str): prompt to generate from
                config (dict): contains temperature, number of tokens parameters

            Returns:
                str
        """
        model_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=2048).to("cuda")
        print(*model_inputs, flush=True)
        print(model_inputs.input_ids, flush=True)
        
        gen_tokens = model.generate(
            **model_inputs,
            do_sample=config.do_sample,
            temperature=config.temperature,
            max_length= config.max_length,
            max_new_tokens=config.max_new_tokens
            # max_new_tokens=500
        )
        print(gen_tokens, flush=True)
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return gen_text


class PhindGeneration(Generation):
    def generate(self, text : str, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, config : InferenceConfig) -> str:
        model_inputs = tokenizer(text, return_tensors='pt', padding=True).to("cuda")
        gen_tokens = model.generate(
            **model_inputs,
            do_sample=config.do_sample,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            max_length= config.max_length,
            max_new_tokens=config.max_new_tokens,
        )
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return gen_text