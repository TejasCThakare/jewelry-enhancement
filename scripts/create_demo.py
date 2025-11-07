import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def create_demo():
    """Create and launch Gradio demo."""
    print("Creating Gradio demo...")
    print("This will import the demo app and launch it.")
    print()
    
    # Import and run demo
    from demo.app import demo
    
    print("Launching demo...")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


def main():
    """Main function."""
    create_demo()


if __name__ == "__main__":
    main()
