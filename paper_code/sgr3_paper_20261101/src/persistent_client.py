"""One vLLM engine and one processor for generation, matching and all scans."""
from local_vllm_client import LocalVLLMClient
from local_matching_client import LocalMatchingClient
_ENGINE=None
_PROCESSOR=None
_SAMPLING=None
class _PersistentClient:
 def __init__(self,config):
  global _ENGINE,_PROCESSOR,_SAMPLING
  self.config=dict(config)
  if _ENGINE is None:
   first=LocalMatchingClient(config)
   _ENGINE,_PROCESSOR,_SAMPLING=first.llm,first.processor,first.SamplingParams
  self.llm,self.processor,self.SamplingParams=_ENGINE,_PROCESSOR,_SAMPLING
class GenerationClient(_PersistentClient):
 complete=LocalVLLMClient.complete
class MatchingClient(_PersistentClient):
 def __init__(self,config):
  super().__init__(config)
  self.config.update(model='Qwen3-VL-32B-Instruct-local-vllm',base_url='local://vllm')
 complete=LocalMatchingClient.complete
