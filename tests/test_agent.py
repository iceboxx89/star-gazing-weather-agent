"""Unit tests for the agent loop: run() driven by a fake model.

The fake speaks litellm's real response types (ModelResponse/Choices/Message),
so the machine under test is byte-identical to production — only the network
and the model's free will are removed. Deterministic, instant, rate-limit-free.
"""

import asyncio
import json
from typing import Any, Callable

import pytest
from litellm import ModelResponse
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Message

import star_gazing_weather_agent.agent as ask
from star_gazing_weather_agent.agent import ask_agent, run
from star_gazing_weather_agent.messages import ChatMessage


class FakeModel:
    """Model function that replays a script of Messages, one per call.

    Records call_count and the wire transcript of every request, so tests can
    assert the loop exits when it should and that observations made it back.
    """

    def __init__(self, script: list[Message]) -> None:
        self.script = script
        self.call_count = 0
        self.requests: list[list[dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> ModelResponse:
        self.call_count += 1
        self.requests.append(kwargs["messages"])
        return ModelResponse(
            choices=[Choices(finish_reason="stop", index=0, message=self.script.pop(0))]
        )


def fake_model_with(script: list[Message]) -> FakeModel:
    """A scripted FakeModel: replay these Messages, one per loop iteration."""
    return FakeModel(script)


def tool_call_message(name: str, arguments: str) -> Message:
    """An assistant reply that requests exactly one tool call."""
    return Message(
        content=None,
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function={"name": name, "arguments": arguments},
            )
        ],
        function_call=None,
    )


def tool_calls_message(
    calls: list[tuple[str, str, str]],
) -> Message:
    """An assistant reply requesting several tool calls: (id, name, arguments)."""
    return Message(
        content=None,
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=call_id,
                type="function",
                function={"name": name, "arguments": arguments},
            )
            for call_id, name, arguments in calls
        ],
        function_call=None,
    )


def answer_message(content: str) -> Message:
    """A final assistant reply: no tool calls."""
    return Message(
        content=content, role="assistant", tool_calls=None, function_call=None
    )


def test_run_dispatch_and_observe() -> None:
    """Real registry: an async tool is awaited through the loop, its result
    (here an out-of-range error, which fires before any network I/O) chains
    back to the call that requested it."""
    messages = [ChatMessage(role="user", content="forecast near the moon")]
    fake = fake_model_with(
        [
            tool_call_message("get_forecast", '{"lat": 999, "lon": 0}'),
            answer_message("final answer"),
        ]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=3))

    assert answer == "final answer"
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]

    # Narration filter: the assistant entry carries tool_calls but no content.
    assert messages[1].content is None
    assert messages[1].tool_calls is not None
    assert messages[1].tool_calls[0]["function"]["name"] == "get_forecast"

    # Observation: the tool result chains to the call id that requested it.
    assert messages[2].tool_call_id == "call_1"
    assert messages[2].content is not None
    assert "coordinates out of range" in messages[2].content


def test_happy_path_two_tool_calls_then_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: two tool calls in one reply, both dispatched, prose out.

    Tools are faked too: the registry is swapped for spy functions registered
    only for this test, so nothing in tools.py runs.
    """
    forecast = {"site": "Milton Keynes", "cloud": 15}
    moon = {"date": "2026-09-16", "illumination": 60}
    tool_calls_made: list[dict[str, Any]] = []

    def fake_forecast(**kwargs: Any) -> dict[str, Any]:
        tool_calls_made.append(kwargs)
        return forecast

    def fake_moon(**kwargs: Any) -> dict[str, Any]:
        tool_calls_made.append(kwargs)
        return moon

    monkeypatch.setattr(
        ask,
        "TOOL_REGISTRY",
        {"get_forecast": fake_forecast, "get_moon_phase": fake_moon},
    )

    messages = [ChatMessage(role="user", content="forecast and moon for tonight")]
    fake = fake_model_with(
        [
            tool_calls_message(
                [
                    ("call_1", "get_forecast", '{"site": "Milton Keynes"}'),
                    ("call_2", "get_moon_phase", '{"date": "2026-09-16"}'),
                ]
            ),
            answer_message("Tonight fits: clear skies, moon low."),
        ]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=3))

    assert answer == "Tonight fits: clear skies, moon low."
    # Both tools ran, in request order, with parsed arguments.
    assert tool_calls_made == [{"site": "Milton Keynes"}, {"date": "2026-09-16"}]
    # Transcript: user -> assistant(calls) -> tool x2 -> assistant(answer).
    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    # Each result chains to the call id that asked for it.
    assert messages[1].tool_calls is not None
    assert [tc["id"] for tc in messages[1].tool_calls] == ["call_1", "call_2"]
    assert messages[2].tool_call_id == "call_1"
    assert messages[3].tool_call_id == "call_2"
    # The information really went back: the second model request carries both
    # tool results in wire format, keyed to their calls.
    assert fake.requests[1] == [
        {"role": "user", "content": "forecast and moon for tonight"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_forecast",
                        "arguments": '{"site": "Milton Keynes"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_moon_phase",
                        "arguments": '{"date": "2026-09-16"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(forecast)},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(moon)},
    ]


def test_help_intent_returns_usage_text() -> None:
    """A how-do-I-use-this question is answered by the gate with usage text —
    no tools, one model call."""
    messages = [ChatMessage(role="user", content="how do i use this tool?")]
    fake = fake_model_with([answer_message("HELP")])

    answer = asyncio.run(ask_agent("how do i use this tool?", model_fn=fake))

    assert answer == ask.HELP_MESSAGE
    assert fake.call_count == 1


def test_unknown_intent_reply_fails_closed() -> None:
    """A classifier reply that matches no token refuses by default: off-topic
    and ambiguous questions both get the refusal, one model call."""
    fake = fake_model_with([answer_message("sure, ask me anything!")])

    answer = asyncio.run(ask_agent("anything goes?", model_fn=fake))

    assert answer == ask.REFUSAL_MESSAGE
    assert fake.call_count == 1


def test_other_intent_returns_refusal() -> None:
    """An off-topic question is refused by the gate itself: one model call,
    deterministic refusal, no loop, no tools."""
    fake = fake_model_with([answer_message("OTHER")])

    answer = asyncio.run(ask_agent("what is the capital of France?", model_fn=fake))

    assert answer == ask.REFUSAL_MESSAGE
    assert fake.call_count == 1


def test_observe_intent_runs_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """An observing question passes the gate and runs the loop to a grounded
    answer. The gate consumes OBSERVE, then the loop runs tool call + answer."""
    monkeypatch.setattr(ask, "TOOL_REGISTRY", {})
    fake = fake_model_with(
        [
            answer_message("OBSERVE"),
            tool_call_message("get_forecast", "{}"),
            answer_message("Clear skies tonight."),
        ]
    )

    answer = asyncio.run(ask_agent("is tonight good at Mauna Kea?", model_fn=fake))

    assert answer == "Clear skies tonight."
    assert fake.call_count == 3


def test_non_observing_question_is_refused() -> None:
    """A tool-less reply carrying the refusal prefix ends the loop at once:
    the model declined rather than dodged, so no steering or tools are needed."""
    messages = [ChatMessage(role="user", content="what is the capital of France?")]
    fake = fake_model_with(
        [answer_message("REFUSED: I only answer observing questions.")]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=3))

    assert answer == "REFUSED: I only answer observing questions."
    assert fake.call_count == 1
    assert [m.role for m in messages] == ["user", "assistant"]


def test_tool_less_reply_is_steered_back_to_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reply with no tool calls before any tool ran is not the answer.

    The model is sent back with a nudge to use its tools; only a reply that
    follows a tool observation terminates the loop. Empty registry: the forced
    forecast call fails as "tool not found", still an observation.
    """
    monkeypatch.setattr(ask, "TOOL_REGISTRY", {})
    messages = [ChatMessage(role="user", content="any news?")]
    fake = fake_model_with(
        [
            answer_message("Nope, all quiet."),
            tool_call_message("get_forecast", "{}"),
            answer_message("Forecast says clear."),
        ]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=5))

    assert answer == "Forecast says clear."
    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert fake.call_count == 3
    # The nudge reached the model: the second request ends with the steering
    # message, and the third is already grounded in the failed tool call.
    assert "did not call any of your tools" in fake.requests[1][-1]["content"]


def test_run_unknown_tool_reports_error() -> None:
    messages = [ChatMessage(role="user", content="forecast")]
    fake = fake_model_with(
        [
            tool_call_message("no_such_tool", "{}"),
            answer_message("recovered"),
        ]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=3))

    assert answer == "recovered"
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[2].content is not None
    assert "tool not found: no_such_tool" in messages[2].content
    assert messages[3].content == "recovered"


def test_run_hits_iteration_cap() -> None:
    messages = [ChatMessage(role="user", content="forecast")]
    fake = fake_model_with(
        [
            tool_call_message("get_forecast", "{}"),
            tool_call_message("get_forecast", "{}"),
        ]
    )

    answer = asyncio.run(run(messages, model_fn=fake, max_iterations=2))

    assert answer is None
    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
