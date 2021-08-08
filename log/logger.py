import getpass
import logging
import os
import sys

from datetime import datetime


class Logger:
    def __new__(cls):
        return logging.basicConfig(level=logging.INFO,
                                    format='%(asctime)s %(levelname)s: %(message)s',
                                   handlers=[
                                       logging.FileHandler(os.path.join('log',
                                                          f'log_{getpass.getuser()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                                       logging.StreamHandler(sys.stdout)
                                   ]
                                   )
