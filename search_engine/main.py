import sys
import uuid
import argparse

from document_retriever import DocumentRetriever
from data_builder import DataBuilder
from data_trainer import DataTrainer
from profile_builder import Profile, interest_integrater


def main():
    dt = DataTrainer()
    db = DataBuilder()
    dr = DocumentRetriever()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", help="It scraps the dataset and retrains the machine learning models.")
    parser.add_argument("--retrain", help="It retrains the machine learning models.")
    parser.add_argument("--newuser", help="You can add a new user and new interests.")
    args = parser.parse_args()

    with open('welcome', 'r') as f:
        msg = f.read()
        print(msg)

    # to crawl documents about the topics in user profiles
    if args.rebuild:
        db.build_training_data()
        dt.train_documents()

    if args.retrain:
        dt.train_documents()

    if args.newuser:
        interests = input("Enter your interests. Please divide by comma(,). :")
        interests = [x.strip() for x in interests.split(',')]
        db.build_training_data(new_topics=interests)
        dt.train_documents()
        new_profile = Profile(name='user{}'.format(uuid.uuid4()), interest=interests)
        interest_integrater.insert_profile(new_profile)

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
