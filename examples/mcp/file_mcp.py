import argparse
import sys
from camel.toolkits import FileWriteToolkit


def get_toolkit():
    file_toolkit = FileWriteToolkit()
    return file_toolkit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Arxiv Toolkit with MCP server mode.",
        usage=f"python {sys.argv[0]} [--mode MODE]",
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP server mode (default: 'stdio')",
    )

    args = parser.parse_args()
    file_toolkit = get_toolkit()
    file_toolkit.mcp.run(args.mode)
