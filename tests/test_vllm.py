import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import vllm
from vllm import LLM, SamplingParams
import time

pretrain_path = 'OpenRLHF/Llama-3-8b-sft-mixture'
prompt = "The quick brown fox jumps over the lazy dog."
TEST_TIME = 2
def test_vllm():
    # 使用vllm加载pretrain model 在GPU0
    v_llm = vllm.LLM(pretrain_path)
    # 使用普通方法加载pretrain model 在GPU1
    # llm = AutoModelForCausalLM.from_pretrained(pretrain_path).to('cuda:1')
    # tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    # 同一个prompt，生成相同长度的response，对比时间消耗
    # 统计时间消耗
    start = time.time()
    sampling_params = SamplingParams(max_tokens=1024)
    for i in range(TEST_TIME):
        outputs = v_llm.generate(prompt, sampling_params)
    print('VLLM time:', time.time() - start)
    print('VLLM response: ', outputs[0].outputs[0].text)
    print('VLLM resp length: ', len(outputs[0].outputs[0].token_ids))
    start = time.time()
    # llm_res = llm.generate(**tokenizer(prompt, return_tensors='pt'), max_length=1024)
    print('LLM time:', time.time() - start)
    # LM time: 2.384185791015625e-07

def test_llm():
    # 使用普通方法加载pretrain model 在GPU1
    llm = AutoModelForCausalLM.from_pretrained(pretrain_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    # 同一个prompt，生成相同长度的response，对比时间消耗
    # 统计时间消耗
    start = time.time()
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    for i in range(TEST_TIME):
        llm_res_tokens = llm.generate(**inputs, max_length=1024)
    llm_res = tokenizer.decode(llm_res_tokens[0])
    print('VLLM time:', time.time() - start)
    print('VLLM response: ', llm_res)
    print('VLLM resp length: ', len(llm_res_tokens[0]))
    # 29.778308391571045
    print('LLM time:', time.time() - start)


if __name__ == '__main__':
    # main()
    test_llm()
