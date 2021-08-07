from pipeline.pipeline import Pipeline
from pipeline.argparser import initialize_parser


if __name__ == '__main__':

    args = initialize_parser()
    pipeline = Pipeline(args)
    if len(pipeline.images) > 0:
        pipeline.run()
