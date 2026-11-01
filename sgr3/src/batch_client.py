"""Up to4 independent requests, serialized into single-engine vLLM batches."""
import json,queue,threading,time,hashlib
from concurrent.futures import Future
from persistent_client import GenerationClient
import persistent_client as pool
from local_vllm_client import prepare_qwen_vllm
_BROKER=None
class Broker:
 def __init__(self):
  GenerationClient({'temperature':0,'max_tokens':1536})
  self.llm,self.processor,self.sampling=pool._ENGINE,pool._PROCESSOR,pool._SAMPLING
  self.queue=queue.Queue();self.batches=0
  # Only this thread calls generate after initialization; no concurrent engine use.
  self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
 def work(self):
  while True:
   first=self.queue.get();items=[first]
   until=time.monotonic()+.03
   while len(items)<8 and time.monotonic()<until:
    try:items.append(self.queue.get(timeout=max(.0001,until-time.monotonic())))
    except queue.Empty:break
   try:
    max_tokens=max(x[1] for x in items);requests=[prepare_qwen_vllm(x[0],self.processor,x[1])[0] for x in items];start=time.monotonic();outputs=self.llm.generate(requests,[self.sampling(temperature=0.0,max_tokens=x[1],top_k=-1,stop=['<END_JSON>']) for x in items],use_tqdm=False);elapsed=time.monotonic()-start;assert len(outputs)==len(items);self.batches+=1
    for (messages,budget,future),output in zip(items,outputs):
     raw=output.outputs[0];text=raw.text
     for i,c in enumerate(text):
      if c!='{':continue
      try:
       obj,_=json.JSONDecoder().raw_decode(text[i:])
       if isinstance(obj,dict):text=json.dumps(obj,ensure_ascii=False);break
      except (ValueError,TypeError):pass
     future.set_result({'text':text,'raw_text':raw.text,'finish_reason':raw.finish_reason,'requested_model':'Qwen/Qwen3-VL-32B-Instruct','returned_model':'Qwen3-VL-32B-Instruct-local','backend':'vllm0.11.2','seconds':elapsed,'batch_size':len(items),'usage':{'prompt_tokens':len(output.prompt_token_ids),'completion_tokens':len(raw.token_ids)},'request_parameters':{'temperature':0,'enable_thinking':False,'max_tokens':budget,'stop':['<END_JSON>']},'request_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()})
   except BaseException as error:
    for _,_,future in items:future.set_exception(error)
 def complete(self,messages,budget):
  future=Future();self.queue.put((messages,budget,future));return future.result()
def initialize():
 global _BROKER
 assert _BROKER is None;_BROKER=Broker();return _BROKER
class LocalBatchClient:
 def __init__(self,config):self.config=dict(config);assert _BROKER is not None
 def complete(self,messages):return _BROKER.complete(messages,self.config['max_tokens'])
class LocalBatchMatchingClient(LocalBatchClient):
 def __init__(self,config):
  super().__init__(config);self.config.update(model='Qwen3-VL-32B-Instruct-local-vllm',base_url='local://vllm')
