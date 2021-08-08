import argparse


def initialize_parser():
    '''
    Catch the arguments which are added to starting the script from command line.
    :return: class args with each argument as attribute
    '''
    parser = argparse.ArgumentParser(description='Digit Recognizer')
    parser.add_argument('--csv',
        action='store_true',
        help='Return prediction as csv file in test_samples/predicted_samples directory.')

    parser.add_argument('--showpicture',
        action='store_true',
        help='Return prediction as copy of the image with bounding boxes and prediction in title in test_samples/predicted_samples directory.')

    parser.add_argument('--download_model',
        action='store_true',
        help='Return prediction as csv file in test_samples directory.')

    args = parser.parse_args()

    return args
