import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analyzer import BeliefAnalyzer


class MockModelProvider:
    def classify_belief(self, text: str, labels: list[str], multi_label: bool = False) -> dict:
        # ignore multi-label for now
        return {"label": "self_efficacy", "score": 0.9, "all_scores": {l: 0.1 for l in labels}}

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
def client():
    from app import main
    main.storage = MockStorage()
    main.risk_storage = MockGenericStorage()
    main.sentiment_storage = MockGenericStorage()
    main.models = MockModelProvider()
    main.analyzer = BeliefAnalyzer(main.models, main.storage, main.risk_storage, main.sentiment_storage)
    return TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "messages_list": [
            {
                "ref_conversation_id": 100,
                "ref_user_id": 50,
                "transaction_datetime_utc": "2023-10-01T10:00:00Z",
                "screen_name": "TestUser",
                "message": "I believe in simplicity.",
            }
        ]
    }


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestEvaluateBeliefsEndpoint:
    def test_returns_200(self, client, sample_payload):
        response = client.post("/api/v1/evaluate-beliefs", json=sample_payload)
        assert response.status_code == 200

    def test_returns_beliefs(self, client, sample_payload):
        response = client.post("/api/v1/evaluate-beliefs", json=sample_payload)
        data = response.json()
        assert "beliefs" in data
        assert "user_id" in data
        assert data["user_id"] == 50

    def test_returns_downstream_outputs(self, client, sample_payload):
        response = client.post("/api/v1/evaluate-beliefs", json=sample_payload)
        data = response.json()
        assert "downstream_outputs" in data


class TestHistoryEndpoint:
    def test_returns_empty_history_for_new_user(self, client):
        response = client.get("/api/v1/history/999")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 999
        assert data["history"] == []

    def test_returns_history_after_analysis(self, client, sample_payload):
        client.post("/api/v1/evaluate-beliefs", json=sample_payload)
        response = client.get("/api/v1/history/50")
        data = response.json()
        assert data["entry_count"] == 1
