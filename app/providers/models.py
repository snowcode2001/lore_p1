class LocalModelProvider:
    """Runs HuggingFace models locally."""

    def __init__(self):
        self._classifier = None
        self._embedder = None
        self._sentiment_grader = None

    def load_models(self):
        """Preload models into memory."""
        _ = self.classifier
        _ = self.embedder
        _ = self.sentiment_grader

    @property
    def classifier(self):
        if self._classifier is None:
            from transformers import pipeline
            self._classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
        return self._classifier

    @property
    def sentiment_grader(self):
        if self._sentiment_grader is None:
            from transformers import pipeline # probably already cached
            self._sentiment_grader = pipeline( # scoring is decent, but need to stay lightweight
                "sentiment-analysis", 
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                return_all_scores=True,
            )
        return self._sentiment_grader

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def classify_belief(self, text: str, labels: list[str], multi_label: bool = False) -> dict:
        """
        Classify a single belief sentence into one of the provided labels.
        
        Input:
            text: single sentence extracted from a message in l_conv.json
            labels: category list from analyzer.py
            multi_label: if True, scores are independent (can have multiple high scores)
        
        Output: {"label": str, "score": float, "all_scores": dict}
        """
        result = self.classifier(text, labels, multi_label=multi_label)
        return {
            "label": result["labels"][0],
            "score": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"])),
        }
    
    def score_sentiment(self, text: str) -> float:
        """
        Scores sentiment of a sentence or text from 0-1.

        Input:
            text: a single message from the conversation
        
        Output: a score from positive likelihood - negative likelihood
        """
        scores = self.sentiment_grader(text)[0]
        scores = {t["label"]: t["score"] for t in scores}
        return scores["positive"] - scores["negative"]


    def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single belief sentence.
        
        Input:
            text: single sentence extracted from a message in l_conv.json
        
        Output: list of 384 floats
        """
        embedding = self.embedder.encode(text)
        return embedding.tolist()
