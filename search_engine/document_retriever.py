from profile_builder import interest_integrater, Profile
from data_trainer import DataTrainer


class DocumentRetriever:
    def match_document_with_user(self, doc: str) -> Profile:
        dt = DataTrainer()
        predicted_interest = dt.predict_interest(doc)
        return predicted_interest
