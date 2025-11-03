#!/usr/bin/env python3
"""
Demonstrate Claude Agent SDK usage and token consumption tracking.

This script shows:
1. Basic query usage (single interaction)
2. Multi-turn conversation with ClaudeSDKClient
3. Custom tool creation
4. Token usage tracking per turn

Requirements:
    pip install claude-agent-sdk
    npm install -g @anthropic-ai/claude-code
"""

import asyncio
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions, tool, create_sdk_mcp_server

load_dotenv()

from utils.step_logger import StepLogger


class MessageTurn(BaseModel):
    """Track a single turn in the conversation"""
    turn_number: int
    prompt: str
    response: str
    message_count: int  # Number of messages streamed in this turn


class ConversationHistory(BaseModel):
    """Complete conversation history with all turns"""
    turns: List[MessageTurn]
    total_turns: int


# Custom tool example
@tool(
    name="calculate_fibonacci",
    description="Calculate the nth Fibonacci number",
    input_schema={
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "The position in the Fibonacci sequence (must be positive)"
            }
        },
        "required": ["n"]
    }
)
async def calculate_fibonacci(args):
    """Calculate the nth Fibonacci number."""
    n = args["n"]
    if n <= 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        result = b

    return {
        "content": [
            {"type": "text", "text": str(result)}
        ]
    }


async def demo_basic_query(logger: StepLogger):
    """Demonstrate basic one-off query usage"""
    logger.step("Demo 1: Basic Query", inputs={
        "description": "Single interaction with no session memory",
        "prompt": "What is 2 + 2?"
    })

    print("\n" + "="*60)
    print("DEMO 1: Basic Query (Single Interaction)")
    print("="*60)

    messages = []
    message_count = 0

    async for message in query(prompt="What is 2 + 2? Give me just the answer."):
        print(f"📨 Message {message_count + 1}: {message}")
        messages.append(message)
        message_count += 1

    logger.output({
        "messages_received": message_count,
        "messages": messages,
        "note": "SDK prints token usage to stderr automatically"
    })

    print(f"\n✅ Received {message_count} messages")
    print("💡 Token usage printed to stderr by SDK")


async def demo_multi_turn_conversation(logger: StepLogger):
    """Demonstrate multi-turn conversation with session memory"""
    logger.step("Demo 2: Multi-turn Conversation", inputs={
        "description": "Multiple turns with session memory",
        "turns": [
            "Tell me about Python in one sentence",
            "What language were we just discussing?",  # Tests memory
            "Give me 3 key features of it"
        ]
    })

    print("\n" + "="*60)
    print("DEMO 2: Multi-turn Conversation (Session Memory)")
    print("="*60)

    conversation_turns = []

    # Create a client for multi-turn conversation using context manager
    async with ClaudeSDKClient() as client:
        # Turn 1: Initial question
        print("\n🔹 Turn 1: Tell me about Python in one sentence")
        await client.query("Tell me about Python in one sentence")

        messages = []
        async for message in client.receive_response():
            messages.append(message)

        response = " ".join(str(m) for m in messages)
        print(f"🤖 Response: {response[:200]}...")

        conversation_turns.append(MessageTurn(
            turn_number=1,
            prompt="Tell me about Python in one sentence",
            response=response,
            message_count=len(messages)
        ))

        # Turn 2: Test memory (should know we discussed Python)
        print("\n🔹 Turn 2: What language were we just discussing?")
        await client.query("What language were we just discussing?")

        messages = []
        async for message in client.receive_response():
            messages.append(message)

        response = " ".join(str(m) for m in messages)
        print(f"🤖 Response: {response[:200]}...")

        conversation_turns.append(MessageTurn(
            turn_number=2,
            prompt="What language were we just discussing?",
            response=response,
            message_count=len(messages)
        ))

        # Turn 3: Follow-up question
        print("\n🔹 Turn 3: Give me 3 key features of it")
        await client.query("Give me 3 key features of it")

        messages = []
        async for message in client.receive_response():
            messages.append(message)

        response = " ".join(str(m) for m in messages)
        print(f"🤖 Response: {response[:200]}...")

        conversation_turns.append(MessageTurn(
            turn_number=3,
            prompt="Give me 3 key features of it",
            response=response,
            message_count=len(messages)
        ))

    history = ConversationHistory(
        turns=conversation_turns,
        total_turns=len(conversation_turns)
    )

    logger.output({
        "conversation": history.model_dump(),
        "note": "Each turn's token usage printed to stderr by SDK"
    })

    print(f"\n✅ Completed {len(conversation_turns)} turns")
    print("💡 Notice how Turn 2 shows the agent remembers context from Turn 1")


async def demo_custom_tools(logger: StepLogger):
    """Demonstrate custom tool usage"""
    logger.step("Demo 3: Custom Tool Usage", inputs={
        "description": "Agent using custom Fibonacci calculator tool",
        "prompt": "Calculate the 10th Fibonacci number using the calculate_fibonacci tool"
    })

    print("\n" + "="*60)
    print("DEMO 3: Custom Tool Usage")
    print("="*60)

    # Create MCP server with the custom tool
    print("\n🔧 Creating MCP server with Fibonacci tool...")
    server = create_sdk_mcp_server(
        name="math-tools",
        version="1.0.0",
        tools=[calculate_fibonacci]
    )

    # Configure options to use the custom tool
    options = ClaudeAgentOptions(
        mcp_servers={"math": server},
        allowed_tools=["mcp__math__calculate_fibonacci"]
    )

    # Ask Claude to use the tool
    print("🔹 Asking Claude to calculate the 10th Fibonacci number using the tool...")

    messages = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Calculate the 10th Fibonacci number using the calculate_fibonacci tool. Just tell me the result.")

        async for message in client.receive_response():
            messages.append(message)
            print(f"📨 {message}")

    response = " ".join(str(m) for m in messages)

    logger.output({
        "tool_name": "calculate_fibonacci",
        "response": response,
        "message_count": len(messages),
        "note": "Claude invoked the tool automatically"
    })


async def demo_with_options(logger: StepLogger):
    """Demonstrate using ClaudeAgentOptions for configuration"""
    logger.step("Demo 4: Configuration Options", inputs={
        "description": "Using system prompt and permission settings",
        "system_prompt": "You are a concise math tutor who only answers in one sentence.",
        "permission_mode": "acceptEdits"
    })

    print("\n" + "="*60)
    print("DEMO 4: Configuration with ClaudeAgentOptions")
    print("="*60)

    options = ClaudeAgentOptions(
        system_prompt="You are a concise math tutor who only answers in one sentence.",
        permission_mode='acceptEdits'
    )

    messages = []
    prompt = "Explain what a prime number is"

    print(f"\n🔹 Prompt: {prompt}")
    print("⚙️  System prompt: 'You are a concise math tutor who only answers in one sentence.'")

    async for message in query(prompt=prompt, options=options):
        messages.append(message)

    response = " ".join(str(m) for m in messages)
    print(f"🤖 Response: {response}")

    logger.output({
        "prompt": prompt,
        "response": response,
        "message_count": len(messages),
        "note": "Response should be concise due to system prompt"
    })

    print("\n💡 Notice how the system prompt enforced concise responses")


async def main():
    """Run all demos and track token usage"""
    logger = StepLogger("claude_agent_sdk_demo")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("🤖 Claude Agent SDK Token Usage Demo")
    print("="*60)
    print("\n📊 This script demonstrates:")
    print("  1. Basic query (single interaction)")
    print("  2. Multi-turn conversation (session memory)")
    print("  3. Custom tool usage")
    print("  4. Configuration options")
    print("\n💡 Token usage is printed to stderr by the SDK after each interaction")
    print("="*60)

    try:
        # # Demo 1: Basic query
        # await demo_basic_query(logger)

        # # Demo 2: Multi-turn conversation
        # await demo_multi_turn_conversation(logger)

        # Demo 3: Custom tools
        await demo_custom_tools(logger)

        # Demo 4: Configuration options
        await demo_with_options(logger)

        # Finalize
        logger.finalize()

        print("\n" + "="*60)
        print("✅ All demos completed!")
        print("="*60)
        print(f"\n📁 Step log saved to: cache/claude_agent_sdk_demo_{timestamp}.json")
        print("\n💡 Key Findings about Token Consumption:")
        print("  • The SDK automatically prints token usage to stderr after each turn")
        print("  • Each query() call is a separate API request with its own token count")
        print("  • ClaudeSDKClient maintains session memory, accumulating context tokens")
        print("  • Multi-turn conversations consume more tokens due to context preservation")
        print("  • The SDK handles context compaction automatically near limits")

    except Exception as e:
        logger.fail(e)
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
