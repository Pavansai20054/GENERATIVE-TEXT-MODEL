from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPTGenerator:
    def __init__(self, model_name='gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        # Optimization for GPU
        if torch.cuda.is_available():
            self.model = self.model.half()  # FP16 for faster inference
            torch.backends.cudnn.benchmark = True
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_length=150, temperature=0.7, top_k=50):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)