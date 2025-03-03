from gui import create_interface
from helpers import generate_random_port

def main(port=None, share=False):
    port = port or generate_random_port()
    demo = create_interface()
    demo.launch(
        server_port=port,
        server_name='0.0.0.0',
        share=share,
        allowed_paths=["/tmp", "/content"]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--share", action="store_true", help="Enable public sharing")
    args = parser.parse_args()
    main(args.port, args.share)
