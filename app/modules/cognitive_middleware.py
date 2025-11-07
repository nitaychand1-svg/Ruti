class CognitiveMiddleware:
    @staticmethod
    def process_reasoning(raw, context):
        rationale = raw['text']
        # Simple enhancement
        sentiment = "positive" if "good" in rationale.lower() else "negative"
        return {"rationale": rationale, "sentiment": sentiment, "context": context}

cognitive = CognitiveMiddleware()
