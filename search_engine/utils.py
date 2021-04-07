import os


def save_data(filepath: str, data: list):
    try:
        f = open(filepath, 'w')
    except IOError:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        f = open(filepath, 'w')
    finally:
        for item in data:
            f.write("%s\n" % item)
        f.close()
