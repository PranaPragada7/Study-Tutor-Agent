"""
Multi-Agent Evaluation — pytest suite.

Tests that the Tutor and Resource agents communicate correctly and that
the Resource Agent improves the Tutor Agent's decision-making.

Tests:
  1. Communication flow: Tutor sends requests → Resource responds
  2. Knowledge base growth: Resource Agent accumulates topic data
  3. Strategy negotiation: weakness reports trigger strategy suggestions
  4. Prerequisite detection: identifies foundational gaps
  5. Message bus logging: all communication is recorded

Run with: pytest tests/test_multi_agent.py -v
"""

import sys
import os
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.agent_comm import MessageBus, AgentMessage, MessageType
from agents.resource_agent import ResourceAgent
from agents.tutor_agent import TutorAgent
from utils.student_profile import create_profile, record_quiz_result


@pytest.fixture(autouse=True)
def _temp_data_dir():
    """Redirect data dir to a temp folder for every test."""
    import utils.student_profile as sp
    original = sp.DATA_DIR
    sp.DATA_DIR = tempfile.mkdtemp()
    yield
    sp.DATA_DIR = original


def test_explicit_none_client_forces_offline_mode(monkeypatch):
    """A configured API key must not override the app's offline selection."""
    fake_client = object()
    monkeypatch.setattr("agents.tutor_agent.get_default_client", lambda: fake_client)
    monkeypatch.setattr("agents.resource_agent.get_default_client", lambda: fake_client)

    profile = create_profile("Offline Student", [{"name": "Biology"}])
    bus = MessageBus()
    tutor = TutorAgent(profile, message_bus=bus, client=None)
    resource = ResourceAgent(bus, client=None)

    assert tutor.client is None
    assert resource.client is None
    assert "offline mode" in tutor.chat("What should I review?").lower()


class TestBasicCommunication:
    """Test that messages flow correctly between agents via the bus."""

    def test_material_request_gets_response(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        request = AgentMessage(
            sender="TutorAgent",
            receiver="ResourceAgent",
            msg_type=MessageType.REQUEST_MATERIALS,
            content={
                "topic": "Recursion",
                "course": "CS101",
                "student_level": "beginner",
                "context": "Student confused base case with recursive case",
                "learning_style": "visual",
            },
            priority=3,
        )
        bus.send(request)

        tutor_inbox = bus.receive("TutorAgent")
        assert len(tutor_inbox) == 1
        response = tutor_inbox[0]
        assert response.msg_type == MessageType.PROVIDE_MATERIALS
        assert response.content.get("topic") == "Recursion"
        assert len(response.content.get("materials", [])) > 0

    def test_message_log_records_both_directions(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS,
            {"topic": "Loops", "course": "CS101",
             "student_level": "beginner", "learning_style": "balanced"},
        ))

        log = bus.get_conversation_log()
        assert len(log) == 2
        assert log[0]["type"] == "request_materials"
        assert log[1]["type"] == "provide_materials"


class TestWeaknessReportStrategy:
    """Test that weakness reports trigger strategy suggestions."""

    def test_weakness_triggers_strategy(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REPORT_WEAKNESS,
            {"topic": "Recursion", "course": "CS101",
             "accuracy": 0.15, "attempts": 8},
            priority=3,
        ))

        tutor_inbox = bus.receive("TutorAgent")
        assert len(tutor_inbox) == 1
        response = tutor_inbox[0]
        assert response.msg_type == MessageType.SUGGEST_STRATEGY
        assert response.content.get("strategy") is not None
        assert len(response.content.get("suggestion", "")) > 20


class TestKnowledgeBaseCaching:
    """Test that the Resource Agent builds and reuses a knowledge base."""

    def test_first_request_is_fresh(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS,
            {"topic": "Loops", "course": "CS101",
             "student_level": "beginner", "learning_style": "balanced"},
        ))
        resp = bus.receive("TutorAgent")[0]
        assert resp.content.get("from_cache") is False

    def test_second_request_uses_cache(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        for _ in range(2):
            bus.send(AgentMessage(
                "TutorAgent", "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {"topic": "Loops", "course": "CS101",
                 "student_level": "beginner", "learning_style": "balanced"},
            ))
            bus.receive("TutorAgent")

        # The second response was cached — verify via KB stats
        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS,
            {"topic": "Loops", "course": "CS101",
             "student_level": "beginner", "learning_style": "balanced"},
        ))
        resp = bus.receive("TutorAgent")[0]
        assert resp.content.get("from_cache") is True

    def test_knowledge_base_stats(self):
        bus = MessageBus()
        resource = ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS,
            {"topic": "Loops", "course": "CS101",
             "student_level": "beginner", "learning_style": "balanced"},
        ))
        bus.receive("TutorAgent")

        stats = resource.get_knowledge_base_stats()
        assert stats["topics_researched"] == 1
        assert stats["total_requests"] >= 1

    def test_suggest_materials_for_priority_topics(self):
        bus = MessageBus()
        resource = ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS,
            {"topic": "Loops", "course": "CS101",
             "student_level": "beginner", "learning_style": "balanced"},
        ))
        bus.receive("TutorAgent")

        suggestions = resource.suggest_materials_for_topics(["Loops", "Unknown"])
        assert len(suggestions) == 1
        assert suggestions[0]["topic"] == "Loops"
        assert suggestions[0]["title"]
        assert suggestions[0]["explanation"]
        assert suggestions[0]["rationale"]["requested_by"] == "TutorAgent"
        assert suggestions[0]["rationale"]["produced_by"] == "ResourceAgent"
        assert "why_tutor_requested" in suggestions[0]["rationale"]
        assert "why_shown" in suggestions[0]["rationale"]
        assert "wrong" in suggestions[0]["rationale"]["why_shown"].lower()
        assert "quiz_evidence" in suggestions[0]["rationale"]


class TestWeaknessPatternDetection:
    """Test cross-topic weakness pattern detection."""

    def test_multiple_weak_topics_detected(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        for topic in ["Recursion", "Loops", "Functions", "OOP"]:
            bus.send(AgentMessage(
                "TutorAgent", "ResourceAgent",
                MessageType.REPORT_WEAKNESS,
                {"topic": topic, "course": "CS101",
                 "accuracy": 0.2, "attempts": 5},
            ))
            bus.receive("TutorAgent")

        log = bus.get_conversation_log()
        assert len(log) >= 8  # 4 reports + 4 responses


class TestFullCommunicationFlow:
    """Test a realistic multi-turn interaction."""

    def test_full_flow(self):
        bus = MessageBus()
        resource = ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        # 3 material requests
        for i in range(3):
            bus.send(AgentMessage(
                "TutorAgent", "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {"topic": "Recursion", "course": "CS101",
                 "student_level": "beginner",
                 "context": f"Attempt {i+1}: confused about base case",
                 "learning_style": "visual"},
            ))
            bus.receive("TutorAgent")

        # Weakness report
        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REPORT_WEAKNESS,
            {"topic": "Recursion", "course": "CS101",
             "accuracy": 0.0, "attempts": 3},
        ))
        strategy_response = bus.receive("TutorAgent")[0]

        stats = bus.get_stats()
        kb_stats = resource.get_knowledge_base_stats()

        assert stats["total_messages"] >= 8
        assert kb_stats["topics_researched"] >= 1
        assert strategy_response.content.get("strategy") is not None


class TestMessageBusStats:
    """Test that the message bus provides accurate statistics."""

    def test_stats_accuracy(self):
        bus = MessageBus()
        ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REQUEST_MATERIALS, {"topic": "A"}))
        bus.send(AgentMessage(
            "TutorAgent", "ResourceAgent",
            MessageType.REPORT_WEAKNESS,
            {"topic": "B", "accuracy": 0.3, "attempts": 5}))
        bus.receive("TutorAgent")

        stats = bus.get_stats()
        assert "TutorAgent" in stats["registered_agents"]
        assert "ResourceAgent" in stats["registered_agents"]
        assert stats["total_messages"] >= 4
        assert "request_materials" in stats["by_type"]
        assert "report_weakness" in stats["by_type"]


class TestKnowledgeBaseEviction:
    """Test _MAX_KNOWLEDGE_BASE_SIZE cap in ResourceAgent."""

    def test_eviction_keeps_most_requested(self):
        bus = MessageBus()
        resource = ResourceAgent(bus)
        bus.register_agent("TutorAgent")

        # Force a small cap for testing
        resource._MAX_KNOWLEDGE_BASE_SIZE = 3

        # Add 5 topics
        for i in range(5):
            bus.send(AgentMessage(
                "TutorAgent", "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {"topic": f"Topic_{i}", "course": "CS101",
                 "student_level": "beginner", "learning_style": "balanced"},
            ))
            bus.receive("TutorAgent")

        # Request Topic_0 many times so it has high times_requested
        for _ in range(5):
            bus.send(AgentMessage(
                "TutorAgent", "ResourceAgent",
                MessageType.REQUEST_MATERIALS,
                {"topic": "Topic_0", "course": "CS101",
                 "student_level": "beginner", "learning_style": "balanced"},
            ))
            bus.receive("TutorAgent")

        assert len(resource.knowledge_base) <= 3
        # The most-requested topic should survive eviction. KB keys are
        # lowercased internally so case variants collapse to one row,
        # so we look up by the canonical key. The display name is
        # preserved on the entry.
        assert "topic_0" in resource.knowledge_base
        assert resource.knowledge_base["topic_0"]["display_name"] == "Topic_0"
