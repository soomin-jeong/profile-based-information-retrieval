import sys
import uuid

from document_retriever import DocumentRetriever
from data_builder import DataBuilder
from data_trainer import DataTrainer
from profile_builder import Profile, interest_integrater


def main():
    dt = DataTrainer()
    db = DataBuilder()
    dr = DocumentRetriever()

    with open('welcome', 'r') as f:
        msg = f.read()
        print(msg)

    # to crawl documents about the topics in user profiles
    if len(sys.argv) > 2:
        if '--rebuild' in sys.argv:
            db.build_training_data()
            dt.train_documents()

        if '--retrain' in sys.argv:
            dt.train_documents()

        if '--new-user' in sys.argv:
            interests = input("Enter your interests. Please divide by comma(,). :")
            interests = interests.split(',')
            new_profile = Profile(name='user{}'.format(uuid.uuid4()), interest=interests)
            interest_integrater.insert_profile(new_profile)
            db.build_training_data()
            dt.train_documents()

    doc = input("Enter a document to match users' interests:")

    # to handle when the user runs for the first time without training a model beforehand
    try:
        pred = dr.match_document_with_interset(dt, doc)
    except FileNotFoundError:
        dt.train_documents()
        pred = dr.match_document_with_interset(dt, doc)

    matched_users = dr.match_document_with_user(pred)
    names = ', '.join([each.name for each in matched_users])
    print("\nThe matched user is '{}' with interest in '{}'.".format(names, pred))


if __name__ == '__main__':
    main()
