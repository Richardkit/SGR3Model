import json,time,hashlib
from pathlib import Path
def prepare_qwen_vllm(messages, qwen_proc, max_new_tokens: int=8192):
    from qwen_vl_utils import process_vision_info
    import re as _re, json as _json
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        text_token_ids = qwen_proc.tokenizer.encode(text, add_special_tokens=False)
        text_tokens = len(text_token_ids)
    except Exception:
        text_tokens = -1
    (image_inputs, video_inputs, video_kwargs) = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return ({'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}, text_tokens)

_SHARED=None

class LocalMatchingClient:
 def __init__(self,config):
  self.config=dict(config)
  self.config.update(model='Qwen3-VL-32B-Instruct-local-vllm',base_url='local://vllm')
  self.model_path=(Path(__file__).resolve().parents[1]/'reports/model_path.txt').read_text().strip()
  from transformers import AutoProcessor
  from vllm import LLM,SamplingParams
  self.SamplingParams=SamplingParams
  self.processor=AutoProcessor.from_pretrained(self.model_path,local_files_only=True)
  global _SHARED
  if _SHARED is None:
   _SHARED=LLM(model=self.model_path,mm_encoder_tp_mode='data',tensor_parallel_size=1,enable_expert_parallel=False,gpu_memory_utilization=0.70,max_model_len=100000,enforce_eager=False,limit_mm_per_prompt={'image':80,'video':0})
  self.llm=_SHARED
 def complete(self,messages):
  request,text_tokens=prepare_qwen_vllm(messages,self.processor,self.config['max_tokens']);start=time.monotonic();p=self.SamplingParams(temperature=0.0,max_tokens=self.config['max_tokens'],top_k=-1,stop=['<END_JSON>']);output=self.llm.generate([request],p,use_tqdm=False)[0];r=output.outputs[0]
  text=r.text
  for i,c in enumerate(text):
   if c!='{':continue
   try:
    obj,_=json.JSONDecoder().raw_decode(text[i:])
    if isinstance(obj,dict):text=json.dumps(obj,ensure_ascii=False);break
   except (ValueError,TypeError):pass
  return {'raw_text':r.text,'text':text,'finish_reason':r.finish_reason,'requested_model':'Qwen/Qwen3-VL-32B-Instruct','returned_model':'Qwen3-VL-32B-Instruct-local','backend':'vllm0.11.2','seconds':time.monotonic()-start,'usage':{'prompt_tokens':len(output.prompt_token_ids),'completion_tokens':len(r.token_ids)},'request_parameters':{'temperature':0,'enable_thinking':False,'max_tokens':self.config['max_tokens'],'stop':['<END_JSON>']},'request_sha256':hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()}
