import sys
from document_retriever import DocumentRetriever
from data_builder import DataBuilder
from data_trainer import DataTrainer


def main():
    # to crawl documents about the topics in user profiles
    if len(sys.argv) == 2 and sys.argv[1] == '--rebuild':
        data_builder = DataBuilder()
        data_builder.build_training_data()

    doc = input("Enter a document to match users' interests: ")
    dt = DataTrainer()
    dt.train_documents()
    dr = DocumentRetriever()
    matched_user = dr.match_document_with_user(doc)
    print(matched_user)
    # print("The matched user is [", matched_user.name, "] with interest in", matched_user.interest)


if __name__ == '__main__':
    main()