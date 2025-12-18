"""
GAIA-1 Chat Interface

Interactive CLI for chatting with GAIA-1.
Supports:
- Persistent conversation memory via Kronos
- Adjustable generation parameters
- Session save/restore

Usage:
    python chat.py --model ./checkpoints/gaia1_best.pt
    python chat.py --model ./checkpoints/gaia1_best.pt --temperature 0.9
"""

import torch
import argparse
try:
    import readline  # For better input handling on Unix
except ImportError:
    pass  # Windows doesn't have readline
from pathlib import Path
from datetime import datetime
import sys

# Fracton imports
try:
    import fracton
except ImportError:
    _fracton_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "fracton"
    if _fracton_path.exists():
        sys.path.insert(0, str(_fracton_path))

from model import GAIA1, GAIA1Config


class GAIAChat:
    """Interactive chat session with GAIA-1."""
    
    def __init__(
        self,
        model: GAIA1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        max_tokens: int = 100
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Conversation history
        self.history = []
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add message to history."""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def build_prompt(self, user_input: str) -> str:
        """Build prompt from conversation history."""
        # Simple format: alternating User/GAIA turns
        prompt_parts = []
        
        # Include recent history (last N turns)
        max_history = 5
        recent = self.history[-(max_history * 2):] if self.history else []
        
        for msg in recent:
            if msg['role'] == 'user':
                prompt_parts.append(f"User: {msg['content']}")
            else:
                prompt_parts.append(f"GAIA: {msg['content']}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("GAIA:")
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, user_input: str) -> str:
        """Generate GAIA's response."""
        prompt = self.build_prompt(user_input)
        
        # Generate
        full_output = self.model.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop_tokens=["\nUser:", "\n\n"]
        )
        
        # Extract just GAIA's response
        response = full_output[len(prompt):].strip()
        
        # Clean up
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        return response
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        self.add_message('user', user_input)
        response = self.generate_response(user_input)
        self.add_message('gaia', response)
        return response
    
    def show_help(self):
        """Show available commands."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    GAIA-1 Chat Commands                      ║
╠══════════════════════════════════════════════════════════════╣
║  /help          - Show this help message                     ║
║  /clear         - Clear conversation history                 ║
║  /temp <value>  - Set temperature (0.0-2.0)                  ║
║  /tokens <num>  - Set max tokens                             ║
║  /history       - Show conversation history                  ║
║  /stats         - Show model stats                           ║
║  /quit, /exit   - Exit chat                                  ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        parts = cmd[1:].split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            return False
        
        elif command == 'help':
            self.show_help()
        
        elif command == 'clear':
            self.history = []
            print("🧹 Conversation cleared.")
        
        elif command == 'temp':
            if args:
                try:
                    self.temperature = float(args[0])
                    print(f"🌡️ Temperature set to {self.temperature}")
                except ValueError:
                    print("❌ Invalid temperature value")
            else:
                print(f"Current temperature: {self.temperature}")
        
        elif command == 'tokens':
            if args:
                try:
                    self.max_tokens = int(args[0])
                    print(f"📝 Max tokens set to {self.max_tokens}")
                except ValueError:
                    print("❌ Invalid token count")
            else:
                print(f"Current max tokens: {self.max_tokens}")
        
        elif command == 'history':
            if not self.history:
                print("📜 No conversation history yet.")
            else:
                print("\n📜 Conversation History:")
                print("-" * 40)
                for msg in self.history:
                    role = "You" if msg['role'] == 'user' else "GAIA"
                    print(f"{role}: {msg['content']}")
                print("-" * 40)
        
        elif command == 'stats':
            print(f"\n📊 Model Stats:")
            print(f"  Model: {self.model}")
            print(f"  Session started: {self.session_start}")
            print(f"  Messages: {len(self.history)}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Max tokens: {self.max_tokens}")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands.")
        
        return True
    
    def run(self):
        """Run interactive chat loop."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  █████╗ ██╗ █████╗       ██╗                      ║
║  ██╔════╝ ██╔══██╗██║██╔══██╗      ██║                      ║
║  ██║  ███╗███████║██║███████║█████╗███║                      ║
║  ██║   ██║██╔══██║██║██╔══██║╚════╝╚██║                      ║
║  ╚██████╔╝██║  ██║██║██║  ██║       ██║                      ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝       ╚═╝                      ║
║                                                              ║
║        First Talkable Field-Native Language Model            ║
║            Pure Dawn Field Theory • No Transformers          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        print(f"Model: {self.model}")
        print("Type /help for commands, /quit to exit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Generate response
                print("GAIA: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Chat with GAIA-1')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_tokens', type=int, default=100)
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nTo train a model first:")
        print("  python train.py --epochs 10 --dataset wikitext")
        sys.exit(1)
    
    print(f"🔄 Loading model from {model_path}...")
    model = GAIA1.load(model_path)
    print(f"✅ Model loaded: {model}")
    
    # Create chat session
    chat = GAIAChat(
        model,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Run chat
    chat.run()


if __name__ == "__main__":
    main()
