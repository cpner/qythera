from agent.tools.tool_registry import Tool


class ImageGeneratorTool(Tool):
    name = "image_generator"
    description = "Generate images from text prompts"

    def execute(self, prompt: str = "", **kwargs) -> str:
        return f"Image generation not yet implemented. Prompt received: {prompt}"
