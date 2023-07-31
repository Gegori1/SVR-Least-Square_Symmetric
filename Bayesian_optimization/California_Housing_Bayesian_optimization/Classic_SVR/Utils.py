# =============================================================================
### Importing libraries ###
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import json

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# =============================================================================
### Update JSONLogger not to overwrite and remove date from jsonl ###
# =============================================================================

class newJSONLogger(JSONLogger):
    """JSONLogger class with update method"""	
    def __init__(self, path):
        """append new data to jsonl file if it exists"""
        self._path=None
        super(JSONLogger, self).__init__()
        self._path = path if path[-6:] == ".jsonl" else path + ".jsonl"

    def update(self, event, instance):
        """removes date from JSONLogger"""
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
        
# =============================================================================
### Update BayesianOptimization to load previous results ###
# =============================================================================

class newBayesianOptimization(BayesianOptimization):
    """BayesianOptimization class with load_previous method"""
    def load_previous(self, path):
        """loads previous results from jsonl file"""
        try:
            with open(path, "r") as f:
                for line in f:
                    if len(line) > 0:
                        data = json.loads(line)
                        self._space.register(
                            params=data["params"],
                            target=data["target"],
                        )
        except KeyError:
            raise KeyError(
                "There are at least one non-unique value. "
                "Check that the file is not already loaded using the `res` method."
                )
        
# =============================================================================

    