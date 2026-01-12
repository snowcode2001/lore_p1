import re
from app.providers.models import LocalModelProvider
from app.providers.storage import JSONFileStorage

# Downstream teams should define these categories based on their needs.
# Alternatively, use a validated taxonomy (e.g., Schwartz Values, Moral Foundations).
BELIEF_CATEGORIES = [
    "self_efficacy",
    "core_values",
    "social_beliefs",
    "institutional_trust",
    "technology_stance",
]

RISK_CATEGORIES = [
    "self_harm",
    "violence",
    "depression",
]

# Regex patterns to detect explicit belief statements.
# Only catches explicit markers; misses implicit beliefs.
BELIEF_MARKERS = [
    r"\bi believe\b",
    r"\bi feel\b",
    r"\bi think\b",
    r"\bi value\b",
    r"\bi'm worried\b",
    r"\bi firmly believe\b",
    r"\bi've come to believe\b",
    r"\bwe've become\b",
    r"\bwe are\b",
]

BELIEF_PATTERN = re.compile("|".join(BELIEF_MARKERS), re.IGNORECASE)


class BeliefAnalyzer:
    def __init__(
        self, 
        model_provider: LocalModelProvider, 
        storage: JSONFileStorage, 
        risk_storage: JSONFileStorage, 
        sentiment_storage: JSONFileStorage,
    ):
        self.models = model_provider
        self.storage = storage
        self.risk_storage = risk_storage
        self.sentiment_storage = sentiment_storage

    def extract_user_messages(self, messages: list[dict], bot_user_id: int = 1) -> list[dict]:
        return [m for m in messages if m.get("ref_user_id") != bot_user_id]

    def find_belief_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        belief_sentences = []
        for s in sentences:
            s = s.strip()
            if s and BELIEF_PATTERN.search(s):
                belief_sentences.append(s)
        return belief_sentences

    def analyze_belief(self, text: str) -> dict:
        classification = self.models.classify_belief(text, BELIEF_CATEGORIES)
        embedding = self.models.get_embedding(text)
        return {
            "text": text,
            "category": classification["label"],
            "category_confidence": classification["score"],
            "category_scores": classification["all_scores"],
            "embedding": embedding,
        }

    def analyze_conversation(self, conversation: dict) -> dict:
        messages = conversation.get("messages_list", [])
        user_messages = self.extract_user_messages(messages)

        if not user_messages:
            return {"beliefs": [], "user_id": None}

        user_id = user_messages[0].get("ref_user_id")
        conversation_id = user_messages[0].get("ref_conversation_id")

        beliefs = []
        for i, msg in enumerate(user_messages):
            text = msg.get("message", "")
            belief_sentences = self.find_belief_sentences(text)
            for sentence in belief_sentences:
                belief = self.analyze_belief(sentence)
                belief["source_message_index"] = i
                belief["timestamp"] = msg.get("transaction_datetime_utc")
                beliefs.append(belief)

        # support content recommendation and monitor user beliefs
        history = self.storage.get_history(user_id)
        self.storage.save_beliefs(user_id, beliefs)

        # sentiment to support StoryBot developers
        sentiments = []
        for i, msg in enumerate(user_messages):
            sentiments.append({
                "timestamp": msg.get("transaction_datetime_utc"),
                "sentiment": self.models.score_sentiment(msg.get("message", "")),
                "source_message_index": i,
                "ref_conversation_id": msg.get("ref_conversation_id"),
            })
        self.sentiment_storage.save_generic(user_id, sentiments)

        # risk scores to help scan for high risk cases
        risk_scores = []
        for i, msg in enumerate(user_messages):
            risk_classification = self.models.classify_belief(msg.get("message", ""), RISK_CATEGORIES, multi_label=True)
            risk_scores.append({
                "timestamp": msg.get("transaction_datetime_utc"),
                "risk_scores": risk_classification["all_scores"],
                "source_message_index": i,
                "ref_conversation_id": msg.get("ref_conversation_id"),
            })
        self.risk_storage.save_generic(user_id, risk_scores)

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "beliefs": beliefs,
            "belief_count": len(beliefs),
            "historical_entries": len(history),
            "downstream_outputs": self._format_downstream(beliefs, history),
        }

    def _format_downstream(self, beliefs: list[dict], history: list[dict]) -> dict:
        categories = [b["category"] for b in beliefs]
        dominant_category = max(set(categories), key=categories.count) if categories else None
        self_beliefs = [b for b in beliefs if b["category"] == "self_efficacy"]

        return {
            "value_attribution": {
                "self_beliefs": [{"text": b["text"]} for b in self_beliefs],
                "self_belief_count": len(self_beliefs),
            },
            "storybot": {
                "dominant_theme": dominant_category,
                "themes": list(set(categories)),
            },
            "content_recommendation": {
                "topic_affinities": list(set(categories)),
            },
        }
