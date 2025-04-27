from camel.toolkits import ArxivToolkit
from image_analysis import ImageAnalysisToolkit
import argparse
import sys
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


def get_toolkit():
    base_url = os.getenv("OPENAI_API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model_platform = ModelPlatformType.OPENAI
    model_type = ModelType.GPT_4O
    image_model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0},
        api_key=api_key,
        url=base_url,
    )
    image_toolkit = ImageAnalysisToolkit(image_model)
    return image_toolkit


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
    image_toolkit = get_toolkit()
    image_toolkit.mcp.run(args.mode)