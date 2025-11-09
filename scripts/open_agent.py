#!/usr/bin/env python3
"""
Simple Claude Code-like implementation for Gemini Flash 2.5
Mimics core Claude Code functionality: persistent bash, file ops, conversation memory
Two modes: auto (runs without confirmation), user_confirmation (asks before each tool)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# MODELS & CONFIG
# ============================================================================

class Mode(str, Enum):
    AUTO = "auto"
    USER_CONFIRMATION = "user_confirmation"


class Config(BaseModel):
    mode: Mode = Mode.USER_CONFIRMATION
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    working_dir: str = "."


# ============================================================================
# BASH SESSION (persistent state like Claude Code)
# ============================================================================

class BashSession:
    """Maintains persistent bash session between commands"""
    
    def __init__(self, cwd: str = "."):
        self.cwd = os.path.abspath(cwd)
        self.process = None
        self._start()
    
    def _start(self):
        """Start new bash process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
        
        self.process = subprocess.Popen(
            ['/bin/bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.cwd,
            bufsize=0
        )
        # Set working directory
        self.process.stdin.write(f'cd {self.cwd}\n')
        self.process.stdin.flush()
    
    def execute(self, command: str, timeout: int = 30) -> str:
        """Execute command in persistent session"""
        if not self.process or self.process.poll() is not None:
            self._start()
        
        try:
            # Use a marker to know when output ends
            marker = "<<<END_OF_OUTPUT>>>"
            full_cmd = f'{command}\necho "{marker}"\n'
            
            self.process.stdin.write(full_cmd)
            self.process.stdin.flush()
            
            output_lines = []
            while True:
                line = self.process.stdout.readline()
                if marker in line:
                    break
                output_lines.append(line)
            
            output = ''.join(output_lines).strip()
            return output if output else "Command executed successfully (no output)"
            
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def close(self):
        """Close bash session"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass


# ============================================================================
# TOOLS (like Claude Code's Read, Write, Edit, Bash)
# ============================================================================

class Tools:
    """File and bash tools matching Claude Code's capabilities"""

    def __init__(self, bash_session: BashSession, working_dir: str = "."):
        self.bash = bash_session
        self.working_dir = Path(working_dir).resolve()

    def _validate_path(self, path: str) -> Path:
        """
        Validate path is within working directory, return absolute Path.
        Raises ValueError if path is outside working directory.
        """
        # Resolve to absolute path
        if os.path.isabs(path):
            abs_path = Path(path).resolve()
        else:
            abs_path = (self.working_dir / path).resolve()

        # Ensure it's within working_dir (prevents ../ traversal)
        try:
            abs_path.relative_to(self.working_dir)
            return abs_path
        except ValueError:
            raise ValueError(f"Access denied: {path} is outside working directory")
    
    def read_file(self, path: str) -> str:
        """Read file contents"""
        try:
            abs_path = self._validate_path(path)
            content = abs_path.read_text(encoding='utf-8')
            line_count = len(content.splitlines())
            return f"File: {path} ({line_count} lines)\n\n{content}"
        except ValueError as e:
            return f"Error: {e}"
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, path: str, content: str) -> str:
        """Write/create file"""
        try:
            abs_path = self._validate_path(path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding='utf-8')
            lines = len(content.splitlines())
            return f"Successfully wrote {lines} lines to {path}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Edit file by replacing old_str with new_str"""
        try:
            abs_path = self._validate_path(path)
            content = abs_path.read_text(encoding='utf-8')

            if old_str not in content:
                return f"Error: String not found in {path}"

            count = content.count(old_str)
            if count > 1:
                return f"Error: String appears {count} times in {path}. Must be unique."

            new_content = content.replace(old_str, new_str, 1)
            abs_path.write_text(new_content, encoding='utf-8')

            return f"Successfully edited {path}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    def run_bash(self, command: str) -> str:
        """Execute bash command in persistent session"""
        return self.bash.execute(command)
    
    def list_files(self, path: str = ".") -> str:
        """List files in directory"""
        return self.bash.execute(f"ls -la {path}")


# ============================================================================
# AGENT (main orchestrator)
# ============================================================================

class Agent:
    """Main agent orchestrating LLM + tools like Claude Code"""

    SYSTEM_PROMPT = """You are a helpful coding assistant with access to tools.
You can read files, write files, edit files, and execute bash commands in a persistent session.

When given a task:
1. Think step by step
2. Use tools to accomplish the task
3. Verify your work

Always use the tools available to you. Don't describe what you would do - actually do it using the tools."""

    FUNCTION_DECLARATIONS = [
        {
            'name': 'read_file',
            'description': 'Read the contents of a file',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Path to the file to read'}
                },
                'required': ['path']
            }
        },
        {
            'name': 'write_file',
            'description': 'Write content to a file (creates or overwrites)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Path to the file'},
                    'content': {'type': 'string', 'description': 'Content to write'}
                },
                'required': ['path', 'content']
            }
        },
        {
            'name': 'edit_file',
            'description': 'Edit a file by replacing old_str with new_str (old_str must be unique in file)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Path to file'},
                    'old_str': {'type': 'string', 'description': 'Unique string to replace'},
                    'new_str': {'type': 'string', 'description': 'Replacement string'}
                },
                'required': ['path', 'old_str', 'new_str']
            }
        },
        {
            'name': 'run_bash',
            'description': 'Execute a bash command in persistent session (state maintained between calls)',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'Bash command to execute'}
                },
                'required': ['command']
            }
        },
        {
            'name': 'list_files',
            'description': 'List files in a directory',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Directory path (default: current dir)'}
                }
            }
        }
    ]

    def __init__(self, config: Config):
        self.config = config
        self.bash_session = BashSession(config.working_dir)
        self.tools = Tools(self.bash_session, config.working_dir)

        # Setup Gemini
        api_key = config.api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.tool = types.Tool(function_declarations=self.FUNCTION_DECLARATIONS)

        # Store chat history manually
        self.history = []
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name"""
        tool_map = {
            'read_file': self.tools.read_file,
            'write_file': self.tools.write_file,
            'edit_file': self.tools.edit_file,
            'run_bash': self.tools.run_bash,
            'list_files': self.tools.list_files,
        }
        
        func = tool_map.get(name)
        if not func:
            return f"Error: Unknown tool {name}"
        
        try:
            result = func(**args)
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    def _maybe_ask_approval(self, name: str, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Ask for approval if in confirmation mode. Returns (approved, reason)"""
        if self.config.mode == Mode.AUTO:
            return True, None

        print(f"\n🔧 Tool: {name}")
        print(f"📋 Args: {json.dumps(args, indent=2)}")

        while True:
            response = input("\nApprove? [y/n/quit]: ").lower().strip()
            if response in ['y', 'yes']:
                return True, None
            elif response in ['n', 'no']:
                return False, "User denied"
            elif response in ['q', 'quit']:
                sys.exit(0)
            print("Please enter y, n, or quit")
    
    def run(self, prompt: str) -> str:
        """Main agent loop - send prompt, handle tools, return final response"""

        print(f"\n{'='*60}")
        print(f"💭 User: {prompt}")
        print(f"{'='*60}\n")

        max_iterations = 10
        iteration = 0

        # Add user message to history
        self.history.append(types.Content(
            role='user',
            parts=[types.Part(text=prompt)]
        ))

        while iteration < max_iterations:
            iteration += 1

            # Send to Gemini with chat history
            try:
                config = types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    tools=[self.tool],
                    temperature=1.0
                )

                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=self.history,
                    config=config
                )

                # Check if there are function calls
                if response.candidates:
                    candidate = response.candidates[0]

                    if candidate.content and candidate.content.parts:
                        function_calls = [part for part in candidate.content.parts
                                        if hasattr(part, 'function_call') and part.function_call is not None]

                        if function_calls:
                            # Add model's response to history
                            self.history.append(candidate.content)

                            # Handle function calls
                            function_response_parts = []

                            for part in function_calls:
                                fc = part.function_call
                                tool_name = fc.name
                                tool_args = dict(fc.args)

                                print(f"\n🔧 Tool call: {tool_name}")
                                print(f"📋 Args: {json.dumps(tool_args, indent=2)}")

                                # Ask for approval (auto-approves in AUTO mode)
                                approved, reason = self._maybe_ask_approval(tool_name, tool_args)
                                if not approved:
                                    result = reason or "Tool execution denied"
                                    print(f"❌ Denied\n")
                                else:
                                    result = self._execute_tool(tool_name, tool_args)
                                    print(f"✅ Executed\n")
                                    print(f"📤 Result: {result[:200]}{'...' if len(result) > 200 else ''}\n")

                                function_response_parts.append(types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tool_name,
                                        response={'result': result}
                                    )
                                ))

                            # Add function responses to history
                            self.history.append(types.Content(
                                role='user',
                                parts=function_response_parts
                            ))
                            continue
                        else:
                            # No function calls, get text response
                            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                            response_text = ''.join(text_parts)

                            # Add model's response to history
                            self.history.append(candidate.content)

                            print(f"🤖 Assistant: {response_text}\n")
                            return response_text

                # Fallback: extract text from response
                response_text = response.text if hasattr(response, 'text') else "No response generated"
                print(f"🤖 Assistant: {response_text}\n")
                return response_text

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"❌ {error_msg}\n")
                return error_msg

        return "Max iterations reached"
    
    def close(self):
        """Cleanup resources"""
        self.bash_session.close()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""

    print("=" * 60)
    print("🚀 Simple Agent (Claude Code-like) for Gemini Flash 2.5")
    print("=" * 60)

    # Parse args
    mode = Mode.USER_CONFIRMATION
    working_dir = os.getcwd()

    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--auto':
            mode = Mode.AUTO
        elif arg == '--dir' and i + 1 < len(sys.argv):
            working_dir = sys.argv[i + 1]
            i += 1  # Skip next arg
        elif arg in ['--help', '-h']:
            print("\nUsage: open_agent.py [options]")
            print("\nOptions:")
            print("  --auto           Run in auto mode (no confirmations)")
            print("  --dir <path>     Set working directory (default: current dir)")
            print("  --help, -h       Show this help message")
            print()
            return
        i += 1

    # Validate and convert working_dir to absolute path
    working_dir = os.path.abspath(working_dir)
    if not os.path.isdir(working_dir):
        print(f"❌ Error: Directory does not exist: {working_dir}")
        return

    if mode == Mode.AUTO:
        print("⚡ Running in AUTO mode (no confirmations)")
    else:
        print("🛡️  Running in USER_CONFIRMATION mode")

    print(f"📁 Working directory: {working_dir}")
    print()

    # Create config
    config = Config(
        mode=mode,
        working_dir=working_dir
    )
    
    # Create agent
    try:
        agent = Agent(config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Set GEMINI_API_KEY environment variable")
        return
    
    print("✅ Agent ready!")
    print("💡 Type your request or 'quit' to exit\n")
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Run agent
                agent.run(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted")
                cont = input("Continue? [y/n]: ").lower().strip()
                if cont not in ['y', 'yes']:
                    break
    
    finally:
        agent.close()


if __name__ == "__main__":
    main()