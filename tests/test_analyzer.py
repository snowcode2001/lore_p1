import pytest
from app.analyzer import BeliefAnalyzer, BELIEF_PATTERN
from app.providers.storage import JSONFileStorage
import tempfile
import os


class MockModelProvider:
    """Fast mock for unit tests - no actual model inference."""

    def classify_belief(self, text: str, labels: list[str], multi_label: bool = False) -> dict:
        # ignore multi_label for now
        if "too old" in text.lower():
            return {"label": "self_efficacy", "score": 0.9, "all_scores": {l: 0.1 for l in labels}}
        if "privacy" in text.lower():
            return {"label": "institutional_trust", "score": 0.85, "all_scores": {l: 0.1 for l in labels}}
        return {"label": labels[0], "score": 0.5, "all_scores": {l: 0.1 for l in labels}}

    def get_embedding(self, text: str) -> list[float]:
        return [0.1] * 384

    def score_sentiment(self, text: str):
        if "happy" in text:
            return 0.9
        elif "sad" in text:
            return -0.9
        else:
            return 0


class MockStorage:
    """In-memory storage for tests."""

    def __init__(self):
        self.data: dict[int, list[dict]] = {}

    def save_beliefs(self, user_id: int, beliefs: list[dict]) -> None:
        if user_id not in self.data:
            self.data[user_id] = []
        self.data[user_id].append({"beliefs": beliefs})

    def get_history(self, user_id: int) -> list[dict]:
        return self.data.get(user_id, [])

class MockGenericStorage:
    """In-memory storage for tests."""

    def __init__(self):
        self.data: dict[int, list[dict]] = {}

    def save_generic(self, user_id: int, records: list[dict]) -> None:
        if user_id not in self.data:
            self.data[user_id] = []
        self.data[user_id].append({"records": records})

    def get_history(self, user_id: int) -> list[dict]:
        return self.data.get(user_id, [])


@pytest.fixture
def analyzer():
    storage = MockStorage()
    risk_storage = MockGenericStorage()
    sentiment_storage = MockGenericStorage()
    models = MockModelProvider()
    return BeliefAnalyzer(models, storage, risk_storage, sentiment_storage)


@pytest.fixture
def sample_conversation():
    return {
        "messages_list": [
            {
                "ref_conversation_id": 123,
                "ref_user_id": 42,
                "transaction_datetime_utc": "2023-10-01T10:00:00Z",
                "screen_name": "TestUser",
                "message": "I believe that technology should be simpler.",
            },
            {
                "ref_conversation_id": 123,
                "ref_user_id": 1,
                "transaction_datetime_utc": "2023-10-01T10:05:00Z",
                "screen_name": "StoryBot",
                "message": "That's an interesting perspective!",
            },
            {
                "ref_conversation_id": 123,
                "ref_user_id": 42,
                "transaction_datetime_utc": "2023-10-01T10:10:00Z",
                "screen_name": "TestUser",
                "message": "I feel like I'm too old to learn new tricks.",
            },
        ]
    }


class TestBeliefPatternDetection:
    def test_detects_i_believe(self):
        assert BELIEF_PATTERN.search("I believe this is true")

    def test_detects_i_feel(self):
        assert BELIEF_PATTERN.search("I feel like something is wrong")

    def test_detects_i_firmly_believe(self):
        assert BELIEF_PATTERN.search("I firmly believe in privacy")

    def test_no_match_for_plain_statement(self):
        assert not BELIEF_PATTERN.search("The sky is blue")


class TestBeliefAnalyzer:
    def test_extracts_user_messages_only(self, analyzer, sample_conversation):
        messages = sample_conversation["messages_list"]
        user_msgs = analyzer.extract_user_messages(messages)
        assert len(user_msgs) == 2
        assert all(m["ref_user_id"] != 1 for m in user_msgs)

    def test_finds_belief_sentences(self, analyzer):
        text = "Hello there. I believe technology is complex. It's frustrating."
        sentences = analyzer.find_belief_sentences(text)
        assert len(sentences) == 1
        assert "I believe" in sentences[0]

    def test_analyze_conversation_returns_beliefs(self, analyzer, sample_conversation):
        result = analyzer.analyze_conversation(sample_conversation)
        assert result["user_id"] == 42
        assert result["conversation_id"] == 123
        assert len(result["beliefs"]) == 2

    def test_downstream_outputs_present(self, analyzer, sample_conversation):
        result = analyzer.analyze_conversation(sample_conversation)
        assert "downstream_outputs" in result
        assert "value_attribution" in result["downstream_outputs"]
        assert "storybot" in result["downstream_outputs"]
        assert "content_recommendation" in result["downstream_outputs"]

    def test_saves_to_storage(self, analyzer, sample_conversation):
        analyzer.analyze_conversation(sample_conversation)
        history = analyzer.storage.get_history(42)
        assert len(history) == 1


class TestEmptyConversation:
    def test_handles_empty_messages(self, analyzer):
        result = analyzer.analyze_conversation({"messages_list": []})
        assert result["beliefs"] == []
        assert result["user_id"] is None

    def test_handles_bot_only_messages(self, analyzer):
        conv = {
            "messages_list": [
                {"ref_user_id": 1, "message": "Hello!", "ref_conversation_id": 1, "transaction_datetime_utc": "2023-01-01T00:00:00Z", "screen_name": "Bot"}
            ]
        }
        result = analyzer.analyze_conversation(conv)
        assert result["beliefs"] == []
