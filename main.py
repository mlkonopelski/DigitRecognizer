from pipeline.argparser import initialize_parser
from pipeline.pipeline import Pipeline
from log.logger import Logger

if __name__ == '__main__':

    Logger()
    args = initialize_parser()
    pipeline = Pipeline(args)
    if len(pipeline.images) > 0:
        pipeline.run()
