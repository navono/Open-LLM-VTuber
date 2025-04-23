import json

from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Literal, Union
from openai import NOT_GIVEN

from .types import FormattedTool
from .utils.path import validate_file

DEFAULT_FORMATTED_TOOLS_PATH = (
    Path(__file__).parent / "configs" / "formatted_tools.json"
)


class ToolManager:
    """Tool Manager for managing tools."""

    def __init__(
        self, formatted_tools_path: str | Path = DEFAULT_FORMATTED_TOOLS_PATH
    ) -> None:
        """Initialize the Tool Manager.

        Args:
            formatted_tools_path (str | Path): Path to the formatted tools file.
                Default is './configs/formatted_tools.json'.
        """
        self.formatted_tools_path = validate_file(formatted_tools_path)

        try:
            self.tools: Dict[str, Union[Dict[str, Any], FormattedTool]] = json.loads(
                self.formatted_tools_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            logger.error(f"TM: Failed to decode JSON from {self.formatted_tools_path}")
            self.tools = {}
        except FileNotFoundError:
            logger.error(
                f"TM: Formatted tools file not found at {self.formatted_tools_path}"
            )
            self.tools = {}

        self.__enabled = True

        self._preprocess_tools()

    def _preprocess_tools(self) -> None:
        """Preprocess the tools to standard data structure."""
        processed_tools: Dict[str, FormattedTool] = {}
        for tool_name, tool_info in self.tools.items():
            if isinstance(tool_info, dict):
                input_schema = tool_info.get("input_schema", None)
                related_server = tool_info.get("related_server", None)
                generic_schema = tool_info.get("generic_schema", None)

                if input_schema and related_server:
                    description = "No description available."
                    if generic_schema and isinstance(generic_schema, dict):
                        description = generic_schema.get("description", description)
                    elif isinstance(tool_info, dict):
                        description = tool_info.get("description", description)

                    processed_tools[tool_name] = FormattedTool(
                        input_schema=input_schema,
                        related_server=related_server,
                        generic_schema=generic_schema,
                    )

                else:
                    logger.warning(
                        f"TM: Invalid tool format for '{tool_name}'. "
                        "Expected a dictionary with 'input_schema' and 'related_server' keys. Skipping."
                    )
            elif isinstance(tool_info, FormattedTool):
                processed_tools[tool_name] = tool_info
            else:
                logger.warning(
                    f"TM: Unexpected tool format for '{tool_name}'. Skipping."
                )
        self.tools = processed_tools

    def get_tool(self, tool_name: str) -> FormattedTool | None:
        """Get a tool by its name.

        Args:
            tool_name (str): Name of the tool.

        Returns:
            FormattedTool | None: The tool if found, else None.
        """
        tool = self.tools.get(tool_name)
        if isinstance(tool, FormattedTool):
            return tool
        return None

    def get_all_tools(
        self, mode: Literal["OpenAI", "Claude"] = "OpenAI"
    ) -> Union[List[Dict[str, Any]], type(NOT_GIVEN)]:
        """Get all generic schemas formatted for the specified API mode.

        Args:
            mode (Literal["OpenAI", "Claude"]): Mode to get the schemas for.
                Default is "OpenAI".

        Returns:
            List[Dict[str, Any]]: List of schemas that suit API's requirements.
            NotGiven: If the tool manager is not enabled or mode is invalid.
        """
        if not self.__enabled:
            return NOT_GIVEN

        formatted_tools_list = []
        if mode.upper() == "OPENAI":
            for tool in self.tools.values():
                if isinstance(tool, FormattedTool) and tool.generic_schema:
                    formatted_tools_list.append(tool.generic_schema)
        elif mode.upper() == "CLAUDE":
            for tool_name, tool_data in self.tools.items():
                if isinstance(tool_data, FormattedTool):
                    description = "No description available."
                    if tool_data.generic_schema and isinstance(
                        tool_data.generic_schema, dict
                    ):
                        description = tool_data.generic_schema.get(
                            "description", description
                        )
                    elif isinstance(tool_data.input_schema, dict):
                        description = tool_data.input_schema.get(
                            "description", description
                        )

                    claude_tool = {
                        "name": tool_name,
                        "description": description,
                        "input_schema": tool_data.input_schema,
                    }
                    formatted_tools_list.append(claude_tool)
        else:
            logger.warning(f"TM: Invalid mode '{mode}'. Expected 'OpenAI' or 'Claude'.")
            return NOT_GIVEN

        return formatted_tools_list

    def enable(self) -> None:
        """Enable the tool manager."""
        self.__enabled = True

    def disable(self) -> None:
        """Disable the tool manager."""
        self.__enabled = False
