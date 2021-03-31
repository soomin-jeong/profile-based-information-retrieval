
from document_retriever import DocumentRetriever


def main():
    dr = DocumentRetriever()

    doc = input("Enter a document to match users' interests: ")
    matched_user = dr.match_document_with_user(doc)
    print(matched_user)


if __name__ == '__main__':
    main()