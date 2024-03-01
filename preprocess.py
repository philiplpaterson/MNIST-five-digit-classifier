import numpy as np
import pandas as pd
import os
import csv
from PIL import Image


def preprocess_data(csvfile, image_dir, process_dir, label_file):
    # This function preprocesses the data by getting all the images
    # from the data directory, flattening it,
    # matching it up with its label, and annotating the image
    # in a csv while also saving the flattened numpy vector of
    # the image into the file_handler.
    pd_data = pd.read_csv(label_file, header=None)
    all_images = os.listdir(image_dir)
    header_row = ['Image Name', 'Label']

    with open(csvfile, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header_row)

        for image_name in all_images:
            image_vector = np.asarray(Image.open(image_dir + image_name)).flatten()
            norm_image_vector = image_vector / 255.0
            image_index = int(image_name.split('.')[0]) - 1
            label = pd_data.iloc[image_index, 0]
            process_data_name = process_dir + image_name.split('.')[0] + '.npy'
            file_handler = open(process_data_name, 'wb')
            np.save(file_handler, norm_image_vector)
            row_data = [process_data_name, label]
            writer.writerow(row_data)


def main():
    print("Preprocessing Training Data")
    train_label_file = 'data_progS24/labels/train_label.txt'
    train_indir = 'data_progS24/train_data/'
    train_outdir = 'data_progS24/train_processed/'
    train_csv_file = 'data_progS24/labels/train_anno.csv'

    preprocess_data(csvfile=train_csv_file, image_dir=train_indir, process_dir=train_outdir,
                    label_file=train_label_file)

    print("Preprocessing Testing Data")
    test_label_file = 'data_progS24/labels/test_label.txt'
    test_indir = 'data_progS24/test_data/'
    test_outdir = 'data_progS24/test_processed/'
    test_csv_file = 'data_progS24/labels/test_anno.csv'

    preprocess_data(csvfile=test_csv_file, image_dir=test_indir, process_dir=test_outdir, label_file=test_label_file)


if __name__ == "__main__":
    main()
