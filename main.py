from pipeline.argparser import initialize_parser
from pipeline.pipeline import Pipeline

if __name__ == '__main__':

    args = initialize_parser()
    pipeline = Pipeline(args)
    if len(pipeline.images) > 0:
        pipeline.run()
