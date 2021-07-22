# Nieuzywany

import pickle
from Database_creator import ImageDescriptor
import shutil


if __name__ == "__main__":
    data_filename = 'database.txt'
    file = open(data_filename, 'rb')
    database = pickle.load(file)
    file.close()

    for descriptor in database:
        path = "images\\"+descriptor.query+descriptor.filename
        new_path = "verified\\"+descriptor.query+descriptor.filename
        shutil.move(path, new_path)
