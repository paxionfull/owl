from camel.toolkits import ArxivToolkit
from examples.overwrite_modules.image_analysis import ImageAnalysisToolkit
import argparse
import sys
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from examples.overwrite_modules.rednote_toolkit import RedNoteToolkit


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
    rednote_toolkit = RedNoteToolkit()
    rednote_toolkit.mcp.run(args.mode)
