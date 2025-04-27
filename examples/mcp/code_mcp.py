import argparse
import sys
from examples.overwrite_modules.code_execution import CodeExecutionToolkit


def get_toolkit():
    code_toolkit = CodeExecutionToolkit()
    return code_toolkit


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
    code_toolkit = get_toolkit()
    code_toolkit.mcp.run(args.mode)
