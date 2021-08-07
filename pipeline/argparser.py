import argparse


def initialize_parser():
    #TODO: Write instructions!
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--csv',
        action='store_true',
        help='Return prediction as csv file in test_samples directory.')

    parser.add_argument('--showpicture',
        action='store_true',
        help='Return prediction as csv file in test_samples directory.')

    parser.add_argument('--download_model',
        action='store_true',
        help='Return prediction as csv file in test_samples directory.')
    args = parser.parse_args()

    return args