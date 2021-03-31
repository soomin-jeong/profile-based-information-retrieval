from profile_builder import interest_integrater, Profile


class DocumentRetriever:
    def match_document_with_user(self, doc: str) -> Profile:
        return interest_integrater.profiles[0]
