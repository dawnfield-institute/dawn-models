"""Test multi-corpus trained GAIA-1 generation across domains."""
import torch
from model import GAIA1, GAIA1Config

def generate_with_penalty(model, prompt, max_tokens=50, temperature=1.0, rep_penalty=1.3):
    """Generate with repetition penalty to prevent loops."""
    model.eval()
    
    # Tokenize prompt
    tokens = model.tokenizer.encode(prompt)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get context
            context = generated[-model.config.max_context:]
            input_ids = torch.tensor([context], device=model.config.device)
            
            # Forward pass
            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :]  # Last position
            
            # Apply repetition penalty
            for token_id in set(generated[-30:]):  # Penalize recent tokens
                next_logits[token_id] /= rep_penalty
            
            # Temperature
            next_logits = next_logits / temperature
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            if next_token == model.tokenizer.eos_token_id:
                break
    
    return model.tokenizer.decode(generated)

def main():
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config from checkpoint to match training
    checkpoint = torch.load('checkpoints/adaptive_multi_corpus/gaia1_best.pt', map_location=device)
    config = checkpoint['config']
    model = GAIA1(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f'Loaded model (max_context={config.max_context})')

    # Test prompts from different domains
    prompts = [
        ('Encyclopedic', 'The history of'),
        ('Encyclopedic', 'In mathematics,'),
        ('Conversational', 'Hello, how are'),
        ('Conversational', 'What do you think'),
        ('Narrative', 'Once upon a time'),
        ('Narrative', 'The little girl'),
    ]

    print("\n" + "="*60)
    print("GAIA-1 Multi-Domain Generation Test")
    print("="*60)

    for domain, prompt in prompts:
        print(f'\n--- {domain} ---')
        print(f'Prompt: "{prompt}"')
        
        with torch.no_grad():
            output = generate_with_penalty(model, prompt, max_tokens=50, temperature=1.0, rep_penalty=1.5)
        
        print(f'Generated: {output}')

if __name__ == "__main__":
    main()
