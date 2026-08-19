"""
Agent Communication Protocol — Message Bus for Multi-Agent Interaction

This module provides the in-process bus the agents use to talk to each
other. The Tutor Agent sends requests, the Resource Agent responds with
materials.

Message Types:
  - REQUEST_MATERIALS: Tutor asks Resource Agent for study content
  - PROVIDE_MATERIALS: Resource Agent sends back content
  - REPORT_WEAKNESS: Tutor tells Resource Agent about a struggling topic
  - SUGGEST_STRATEGY: Resource Agent suggests a different teaching approach
  - STATUS_UPDATE: Either agent shares its current state

All messages go through a central MessageBus that logs everything
for evaluation and debugging.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from config import MAX_DISPATCH_DEPTH, MESSAGE_LOG_CAP

logger = logging.getLogger(__name__)


class MessageType(Enum):
    REQUEST_MATERIALS = "request_materials"
    PROVIDE_MATERIALS = "provide_materials"
    REPORT_WEAKNESS = "report_weakness"
    SUGGEST_STRATEGY = "suggest_strategy"
    STATUS_UPDATE = "status_update"


class AgentMessage:
    """A message passed between agents."""

    def __init__(
        self, sender: str, receiver: str, msg_type: MessageType, content: dict, priority: int = 1
    ):
        """
        Args:
            sender: Name of the sending agent (e.g., "TutorAgent")
            receiver: Name of the receiving agent (e.g., "ResourceAgent")
            msg_type: Type of message
            content: The message payload
            priority: 1 (low) to 5 (urgent)
        """
        self.sender = sender
        self.receiver = receiver
        self.msg_type = msg_type
        self.content = content
        self.priority = priority
        # L5: timezone-aware UTC so log-replay and sort-by-timestamp
        # behave correctly across DST transitions and mixed TZs.
        self.timestamp = datetime.now(timezone.utc).isoformat()
        # UUID prevents ID collisions during rapid-fire messaging.
        self.id = uuid.uuid4().hex

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.msg_type.value,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


class MessageBus:
    """
    Central message bus that routes messages between agents.
    Logs all communication for evaluation.

    Thread safety:
      All state (message_log, _inbox, _subscribers, _agents) is protected
      by a re-entrant lock. Callbacks are invoked *outside* the lock so
      a callback is free to call back into ``send`` without deadlocking.
      Readers (``get_conversation_log``, ``get_stats``, ``peek``,
      ``receive``) take a snapshot under the lock so a concurrent
      ``_trim_log`` cannot surface a torn view.
    """

    # Backed by config.* so a deployment can tune via env vars. Kept as
    # class attributes so tests can monkey-patch them (cleaner than
    # mutating module globals).
    _LOG_CAP = MESSAGE_LOG_CAP
    _MAX_DISPATCH_DEPTH = MAX_DISPATCH_DEPTH

    def __init__(self):
        self.message_log: list[dict] = []
        self._subscribers: dict[str, list] = {}  # {agent_name: [callback_fn, ...]}
        self._inbox: dict[str, list[AgentMessage]] = {}  # {agent_name: [messages]}
        self._agents: dict[str, Any] = {}  # {agent_name: instance}
        # Reentrant so nested send() from inside a callback doesn't deadlock.
        self._lock = threading.RLock()
        # Per-thread dispatch-depth counter for the recursion guard.
        self._local = threading.local()

    def register_agent(self, agent_name: str, callback=None, instance: Any = None) -> None:
        """
        Register an agent on the bus.

        Args:
            agent_name: Unique identifier for the agent
            callback: Optional function called when a message arrives for this agent
            instance: Optional reference to the agent object itself, so other
                      agents can look it up by name (used for persistence
                      hand-off, e.g. tutor reads resource_agent.to_dict()).
        """
        with self._lock:
            if agent_name not in self._subscribers:
                self._subscribers[agent_name] = []
                self._inbox[agent_name] = []
            if callback:
                self._subscribers[agent_name].append(callback)
            if instance is not None:
                self._agents[agent_name] = instance

    def get_agent(self, agent_name: str) -> Any:
        """Look up a registered agent instance by name (None if not registered)."""
        with self._lock:
            return self._agents.get(agent_name)

    def _trim_log_locked(self) -> None:
        # Caller holds self._lock. Truncate in place so a reader iterating
        # self.message_log doesn't see the list object swapped out from
        # under them mid-iteration.
        overflow = len(self.message_log) - self._LOG_CAP
        if overflow > 0:
            del self.message_log[:overflow]

    def send(self, message: AgentMessage) -> None:
        """
        Send a message through the bus.
        The message is logged and delivered to the receiver's inbox.

        Callbacks for the receiver are invoked AFTER the lock has been
        released so a callback is free to call ``send`` again without
        deadlocking — and slow (e.g., LLM-bound) callbacks never block
        another thread from appending to the log or peeking stats.
        """
        # Recursion guard per-thread: 1 legitimate request + response is
        # depth 2. A cycle of bad callbacks could keep growing forever.
        depth = getattr(self._local, "depth", 0)
        if depth >= self._MAX_DISPATCH_DEPTH:
            logger.error(
                "MessageBus dispatch depth %d exceeded — dropping message %s from %s "
                "(likely a callback cycle)",
                self._MAX_DISPATCH_DEPTH,
                message.id,
                message.sender,
            )
            return

        callbacks_to_run: list = []
        with self._lock:
            self.message_log.append(message.to_dict())
            self._trim_log_locked()

            receiver = message.receiver
            if receiver in self._inbox:
                self._inbox[receiver].append(message)
                callbacks_to_run = list(self._subscribers.get(receiver, []))

        # Run callbacks outside the lock so a callback's own bus.send()
        # (e.g., ResourceAgent replying to a material request) doesn't
        # have to wait on a lock we're already holding. An LLM-bound
        # callback also no longer freezes stats/peek/log from other threads.
        if not callbacks_to_run:
            return

        self._local.depth = depth + 1
        try:
            for callback in callbacks_to_run:
                try:
                    callback(message)
                except Exception as e:
                    # Record the failure as a log entry so it shows up in
                    # Diagnostics, but don't let it crash the sender.
                    with self._lock:
                        self.message_log.append(
                            {
                                "error": f"Callback failed for {receiver}: {str(e)}",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        self._trim_log_locked()
                    logger.exception(
                        "Callback failed while handling message %s for %s",
                        message.id,
                        receiver,
                    )
        finally:
            self._local.depth = depth

    def receive(self, agent_name: str) -> list[AgentMessage]:
        """
        Get all pending messages for an agent (and clear inbox).
        """
        with self._lock:
            if agent_name not in self._inbox:
                return []
            messages = self._inbox[agent_name]
            self._inbox[agent_name] = []
            return messages

    def peek(self, agent_name: str) -> list[AgentMessage]:
        """Check inbox without clearing it."""
        with self._lock:
            # Snapshot so a concurrent send/receive doesn't mutate the
            # returned list under the caller's feet.
            return list(self._inbox.get(agent_name, []))

    def get_conversation_log(self) -> list[dict]:
        """Get the full message log for evaluation."""
        with self._lock:
            return list(self.message_log)

    def get_stats(self) -> dict:
        """Get communication statistics."""
        with self._lock:
            # Snapshot everything we need under the lock, then compute
            # outside — keeps the critical section short.
            log_snapshot = list(self.message_log)
            agents_snapshot = list(self._subscribers.keys())

        stats = {
            "total_messages": len(log_snapshot),
            "registered_agents": agents_snapshot,
            "by_type": {},
            "by_sender": {},
        }
        for msg in log_snapshot:
            if "type" in msg:
                stats["by_type"][msg["type"]] = stats["by_type"].get(msg["type"], 0) + 1
                stats["by_sender"][msg["sender"]] = stats["by_sender"].get(msg["sender"], 0) + 1
        return stats
