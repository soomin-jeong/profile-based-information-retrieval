from profile_builder import interest_integrater, Profile
from data_trainer import DataTrainer


class DocumentRetriever:
    def match_document_with_interset(self, doc: str) -> str:
        dt = DataTrainer()
        predicted_interest = dt.predict_interest(doc)
        return predicted_interest

    def match_document_with_user(self, pred: str) -> [Profile]:
        return interest_integrater.get_users_interested_in(pred)
