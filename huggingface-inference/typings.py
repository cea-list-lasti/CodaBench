from pydantic import BaseModel

class InferenceConfig(BaseModel):
    max_new_tokens : int
    max_length : int
    temperature : float
    do_sample : bool
    
class BenchmarkConfig(BaseModel):
    prompt_file : str
    batch_size : int
    num_generations : int
