from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPTGenerator:
    def __init__(self, model_name='distilgpt2'):
        """Initialize GPT-2 model with GPU support"""
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        # Enable CUDA kernels if available
        if torch.cuda.is_available():
            self.model = self.model.half()  # Use FP16 for faster inference
            torch.backends.cudnn.benchmark = True
            
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=50):
        """Generate text with GPU acceleration"""
        # Move input to GPU
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Use CUDA graph if available for faster generation
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(), torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length + len(input_ids[0]),
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        else:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length + len(input_ids[0]),
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Explicitly clear GPU cache
    torch.cuda.empty_cache()
    generator = GPTGenerator()
    print(generator.generate_text("The future of AI with GPU acceleration"))