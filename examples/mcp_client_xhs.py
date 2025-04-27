from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
import shutil
import json

# Create server parameters for stdio connection
config = json.load(open("mcp/lenovo_mcp.json"))
lenovo = config['mcpServers']['xhs_mcp']

server_params = StdioServerParameters(
    command=lenovo["command"],
    args=lenovo["args"]
)


async def run_client():
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                # print(tools[0].parameters)
                print(f"Available tools: {tools}")

                res = await session.call_tool("extract_xiaohongshu_history", {"profile_id": "6809b984000000001b0351a4"})
                print("extract_xiaohongshu_history done. ", res)
                
                # Add a small delay to ensure all operations complete
                await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error in run_client: {e}")
        raise


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_client())
    print("run done.")
