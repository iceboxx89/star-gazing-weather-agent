"""Unit tests for the agent layer, driven by a scripted pydantic-ai model.

A FunctionModel replays a script of model responses, so the real Agent, the
real toolset and the real grounding validator all run — only the network and
the model's free will are removed.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from star_gazing_weather_agent.agents import StarGazingWeatherAgent, ask_agent
from star_gazing_weather_agent.constants import HELP_MESSAGE, REFUSAL_MESSAGE
from star_gazing_weather_agent.tools._toolset import observing_tools


def answer(text: str) -> ModelResponse:
    """A final text reply."""
    return ModelResponse(parts=[TextPart(text)])


def call_tool(name: str, args: dict[str, Any]) -> ModelResponse:
    """A reply requesting exactly one tool call."""
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])


def intent(value: str) -> Callable[[AgentInfo], ModelResponse]:
    """A classifier reply: the enum value, via the agent's output tool."""

    def step(info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args={"response": value}
                )
            ]
        )

    return step


ScriptStep = ModelResponse | Callable[[AgentInfo], ModelResponse]


class ScriptedModel:
    """Replay a script of responses, one per model request, recording each call."""

    def __init__(self, script: list[ScriptStep]) -> None:
        self.script = script
        self.call_count = 0
        self.requests: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.call_count += 1
        self.requests.append(messages)
        step = self.script.pop(0)
        if isinstance(step, ModelResponse):
            return step
        return step(info)


def scripted(script: list[ScriptStep]) -> tuple[ScriptedModel, FunctionModel]:
    """A scripted model plus the FunctionModel pydantic-ai drives it through."""
    model = ScriptedModel(script)
    return model, FunctionModel(model)


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    """Every tool observation carried by these messages."""
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_all_observing_tools_are_registered() -> None:
    """Importing the tools package registers every tool on the shared toolset."""
    assert set(observing_tools.tools) == {
        "get_forecast",
        "get_moon_phase",
        "get_todays_date",
    }


def test_tool_call_then_grounded_answer() -> None:
    """A tool runs, its (here failing) observation chains back to the model,
    and the following answer is accepted because a tool has returned."""
    model, fn = scripted(
        [
            call_tool("get_forecast", {"lat": 999, "lon": 0}),
            answer("final answer"),
        ]
    )

    result = asyncio.run(StarGazingWeatherAgent(model=fn).ask("forecast near the moon"))

    assert result == "final answer"
    assert model.call_count == 2
    observed = tool_returns(model.requests[1])
    assert observed
    assert "coordinates out of range" in str(observed[0].content)


def test_two_tool_calls_in_one_reply() -> None:
    """Both tools in a single reply are executed before the next model call."""
    model, fn = scripted(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_moon_phase", args={"date": "2026-09-16"}
                    ),
                    ToolCallPart(tool_name="get_todays_date", args={"location": "UK"}),
                ]
            ),
            answer("Tonight fits: clear skies, moon low."),
        ]
    )

    result = asyncio.run(
        StarGazingWeatherAgent(model=fn).ask("forecast and moon for tonight")
    )

    assert result == "Tonight fits: clear skies, moon low."
    assert model.call_count == 2
    assert {part.tool_name for part in tool_returns(model.requests[1])} == {
        "get_moon_phase",
        "get_todays_date",
    }


def test_tool_less_reply_is_steered_back_to_tools() -> None:
    """A reply with no tool calls before any tool ran is not the answer: the
    model is sent back with a nudge, and only a grounded reply ends the loop."""
    model, fn = scripted(
        [
            answer("Nope, all quiet."),
            call_tool("get_moon_phase", {"date": "2026-09-16"}),
            answer("Moon is 60% lit."),
        ]
    )

    result = asyncio.run(StarGazingWeatherAgent(model=fn).ask("any news?"))

    assert result == "Moon is 60% lit."
    assert model.call_count == 3
    steered = any(
        "did not call any of your tools" in str(part)
        for message in model.requests[1]
        for part in message.parts
    )
    assert steered


def test_refusal_reply_ends_the_loop() -> None:
    """A tool-less reply carrying the refusal prefix is accepted at once: the
    model declined rather than dodged, so no steering is needed."""
    model, fn = scripted([answer("REFUSED: I only answer observing questions.")])

    result = asyncio.run(
        StarGazingWeatherAgent(model=fn).ask("what is the capital of France?")
    )

    assert result == "REFUSED: I only answer observing questions."
    assert model.call_count == 1


def test_iteration_cap_returns_none() -> None:
    """A model that never stops calling tools hits the request limit, and the
    agent reports that as None rather than raising."""
    model, fn = scripted(
        [
            call_tool("get_moon_phase", {"date": "2026-09-16"}),
            call_tool("get_moon_phase", {"date": "2026-09-16"}),
            call_tool("get_moon_phase", {"date": "2026-09-16"}),
        ]
    )

    result = asyncio.run(
        StarGazingWeatherAgent(model=fn, max_iterations=2).ask("forecast")
    )

    assert result is None
    assert model.call_count == 2


def test_help_intent_returns_usage_text() -> None:
    """A how-do-I-use-this question is answered by the gate with usage text —
    no tools, one model call."""
    model, fn = scripted([intent("HELP")])

    result = asyncio.run(ask_agent("how do i use this tool?", model=fn))

    assert result == HELP_MESSAGE
    assert model.call_count == 1


def test_other_intent_returns_refusal() -> None:
    """An off-topic question is refused by the gate itself: one model call,
    deterministic refusal, no loop, no tools."""
    model, fn = scripted([intent("OTHER")])

    result = asyncio.run(ask_agent("what is the capital of France?", model=fn))

    assert result == REFUSAL_MESSAGE
    assert model.call_count == 1


def test_unknown_intent_reply_fails_closed() -> None:
    """A classifier reply that matches no enum value refuses by default:
    off-topic and ambiguous questions both get the refusal, one model call."""
    model, fn = scripted([answer("sure, ask me anything!")])

    result = asyncio.run(ask_agent("anything goes?", model=fn))

    assert result == REFUSAL_MESSAGE
    assert model.call_count == 1


def test_observe_intent_runs_the_loop() -> None:
    """An observing question passes the gate and runs the loop to a grounded
    answer. The gate consumes OBSERVE, then the loop runs tool call + answer."""
    model, fn = scripted(
        [
            intent("OBSERVE"),
            call_tool("get_moon_phase", {"date": "2026-09-16"}),
            answer("Clear skies tonight."),
        ]
    )

    result = asyncio.run(ask_agent("is tonight good at Mauna Kea?", model=fn))

    assert result == "Clear skies tonight."
    assert model.call_count == 3
