import os
from huggingface_hub import login
import argparse
import torch
import warnings
import time
import psutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM,LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
import wandb
Tokentime_sum = 0

class TextDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        begin_time_tk = time.time()
        encoded_input = self.tokenizer.encode_plus(
            prompt, 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )
        end_time_tk = time.time()
        global Tokentime_sum
        Tokentime_sum +=end_time_tk - begin_time_tk
        # print(f"Time of tokenization{end_time_tk - begin_time_tk}")
        input_ids = encoded_input['input_ids'].squeeze(0)  # Remove batch dimension
        # attention_mask = encoded_input['attention_mask'].squeeze(0)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

class Model:
    def __init__(self,model_name='gemma-7b-it',use_gpu="gpu",batch_size=1,data_path='dataset/test-v2.0.json',input_max_length=32,output_max_length=64,quantize=False):
        self.model_name = model_name
        self.device = torch.device("cuda:6" if use_gpu=="gpu" else "cpu")
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.prompts = []
        self.data_path = data_path
        self._load_models()
        self._load_prompts()
        self.print_model_details()
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.quantize_ = quantize
        # self.test_infer()
    def create_dataloader(self):
        dataset = TextDataset(self.prompts, self.tokenizer, max_length=self.input_max_length)  
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  
        return dataloader
    def _load_prompts(self):
        import json
        sum_tokens = 0
        with open(self.data_path,'r',encoding='utf-8') as f:
            data = json.load(f)
            for data_ in data['data']:
                for item in data_['paragraphs']:
                    context = item['context']
                    for qas in item['qas']:
                        question = qas['question']
                        
                        prompt = "Context:" + context + '\n' + "Question:" + question + '\n' + "Answer:\n"
                        # print(prompt)
                        self.prompts.append(prompt)
        self.prompts = self.prompts[:10]     
        for prompt in self.prompts:
            sum_tokens += len(self.tokenizer.encode(prompt, add_special_tokens=True))          
        # print the statistics of the prompts
        print(f'Sum of prompts: {len(self.prompts)}')
        print(f'Average tokens of prompts: {sum_tokens/len(self.prompts)}') 
    def _load_models(self):
        huggingface_proxies ={
            'http':'127.0.0.1:10652',
            'https':'127.0.0.1:10652',
            'ftp':'127.0.0.1:10652'
        }
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if "gemma" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,proxies=huggingface_proxies,padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.pad_token
        elif "llama" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,proxies=huggingface_proxies,padding_side="left")   
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.pad_token
        elif "phi" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,proxies=huggingface_proxies,padding_side='left')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.pad_token
        else:
            raise ValueError("Unsupported model name")
        # 4-bit
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        # 8-bit
        # quant_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_threshold = 0.0
        # )
        if self.device.type == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,proxies=huggingface_proxies,device_map=self.device,torch_dtype=torch.float32)
        elif self.device.type == "cuda":
            # if self.quantize_:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,proxies=huggingface_proxies,device_map=self.device,quantization_config=quant_config,torch_dtype=torch.bfloat16)
            # else:
                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name,proxies=huggingface_proxies,device_map=self.device,torch_dtype=torch.bfloat16)
        else:
            raise ValueError("Unsupported device name")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def print_model_details(self):
        from fvcore.nn import FlopCountAnalysis
        from thop import profile
        from thop import clever_format  
        total_params = sum(p.numel() for p in self.model.parameters())
        input_example = torch.randint(0, self.tokenizer.vocab_size, (1, 128)).to(self.model.device)
        flops, params = profile(self.model.model, inputs=(input_example,))
        flops, params = clever_format([flops, params], "%.3f")

        print(f"Total FLOPS: {flops}")
        print(f"Total number of parameters: {total_params}")
        print('-' * 100)
        
    def test_infer(self):
        dataloader = self.create_dataloader()
        warnings.filterwarnings("ignore")
        list_mem = []  # GPU = GPU_MEMORY  CPU = MEMORY
        list_time = []
        list_Throughput = []
        list_inv_Throughput = []
        from tqdm import tqdm
        for k in tqdm(range(0,5),desc="processing epochs"):
            epoch_tokens = 0
            epoch_time = 0
            epoch_mem = []
            begin_time = time.time()
            for batch_inputs, batch_masks in tqdm(dataloader,desc=f"Epoch {k+1}",leave=False):
                batch_inputs = batch_inputs.to(self.device)
                batch_masks = batch_masks.to(self.device)
                # print(f'batch_masks:{batch_masks}\n')
                outputs = self.model.generate(input_ids=batch_inputs, attention_mask=batch_masks,min_new_tokens=self.output_max_length,max_new_tokens=self.output_max_length,do_sample=False)
                if self.device == 'cpu':
                    mem_info = psutil.Process(os.getpid()).memory_info()
                    epoch_mem.append(mem_info.rss / 1024 / 1024)
                else:
                    epoch_mem.append(torch.cuda.memory_allocated(device=self.device) / 1024 / 1024)
                # gap_time = end_time - begin_time
                # epoch_time += gap_time
                for output in outputs:
                    print('-'*100)
                    text = self.tokenizer.decode(output[self.input_max_length:], skip_special_tokens=False)
                    print(f'Answer:{text}\n')
                    print(f'UNKID:{self.tokenizer.unk_token_id}\n')
                    if "llama" in self.model_name:
                        ls = torch.sum(output[self.input_max_length:]!=self.tokenizer.pad_token_id)
                        mask = output[self.input_max_length:] != self.tokenizer.pad_token_id 
                    elif "gemma" in self.model_name:
                        ls = torch.sum(output[self.input_max_length:]!=self.tokenizer.pad_token_id)
                        mask = output[self.input_max_length:] != self.tokenizer.pad_token_id 
                    else:
                        ls = torch.sum(output[self.input_max_length:]!=self.tokenizer.pad_token_id)
                        mask = output[self.input_max_length:] != self.tokenizer.pad_token_id 
                    output = output[self.input_max_length:][mask]
                    text = self.tokenizer.decode(output, skip_special_tokens=False)
                    # print(f'Answer:{text}\n')
                    # print(text[self.input_max_length:])
                    print(f'lenght:{ls}\n')
                    # token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    # print(len(token_ids))
                    epoch_tokens = epoch_tokens + ls
            end_time = time.time()
            epoch_time = end_time - begin_time
            list_time.append(epoch_time)   
            list_Throughput.append(epoch_tokens / epoch_time)
            list_mem.append(sum(epoch_mem) / len(epoch_mem))
            list_inv_Throughput.append(epoch_time/epoch_tokens)
        

        lambda_outlier = lambda lst: sum(lst)/len(lst)
        average_time = lambda_outlier(list_time)
        average_throughput = lambda_outlier(list_Throughput)
        average_mem = lambda_outlier(list_mem)
        average_inv_throughput = lambda_outlier(list_inv_Throughput)
        print("-"*100)
        print(f"Average Time: {average_time}s")
        print(f"Average Throughput: {average_throughput:.2f}tokens/s")
        print(f'Averge Inverse Throughput: {average_inv_throughput:.2f}s/token')
        if self.device =='cpu':
            print(f"Average Memory Usage: {average_mem}MB")
        else:
            print(f"Average GPU Memory Usage: {average_mem}MB")
        print("-"*100)
        wandb.log({
            "Average Time ": average_time,
            "Average Throughput": average_throughput,
            "Average Memory Usage ": average_mem,
            "Average Inverse Throughput ": average_inv_throughput
        })
        wandb.finish()

       
def parse_args():
    parse = argparse.ArgumentParser(description="Select your arguments")
    parse.add_argument("--model", type=str, default="google/gemma-7b-it", help="Select the type of gemma scale")
    parse.add_argument("--device",type=str, default="gpu", help="Select the device to run the model")
    parse.add_argument("--batch_size",type=int,default="1", help="Select the batch size to run the model")
    parse.add_argument("--input_max_length",type=int,default="32",help="Select the maximum length of input")
    parse.add_argument("--output_max_length",type=int,default="64",help="Select the maximum length of output")
    parse.add_argument("--quantize",type=bool,default=False,help="Select the quantization of the model")
    # print the setting of inference
    print('-' * 100)
    return parse.parse_args()
    
def main():
    # login()
    wandb.login()
    wandb.init(project="Version_4.0-GPU",name='gemma-7b-4bit-batch=1')
    args = parse_args()
    args_dict = vars(args)
    # wandb.config = args_dict
    print("The setting of this inference: ")    
    for arg_name, arg_value in args_dict.items():
        print(f"{arg_name}: {arg_value}")
    model = Model(model_name=args.model,use_gpu=args.device,batch_size=args.batch_size,data_path='dataset/test-v2.0.json',
                  input_max_length=args.input_max_length,output_max_length=args.output_max_length,quantize=args.quantize)
    # begin to test the inference
    model.test_infer()
    
    
if __name__ == "__main__":
    main()
    
