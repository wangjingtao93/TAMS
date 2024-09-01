# load libraries
import numpy as np
import pandas as pd
from miseval import evaluate

# Get some ground truth / annotated segmentations
np.random.seed(1)
real_bi = np.random.randint(2, size=(64,64))  # binary (2 classes)
real_mc = np.random.randint(5, size=(64,64))  # multi-class (5 classes)
# Get some predicted segmentations
np.random.seed(2)
pred_bi = np.random.randint(2, size=(64,64))  # binary (2 classes)
pred_mc = np.random.randint(5, size=(64,64))  # multi-class (5 classes)

# Run binary evaluation
dice = evaluate(real_bi, pred_bi, metric="DSC")    
  # returns single np.float64 e.g. 0.75

# Run multi-class evaluation
dice_list = evaluate(real_mc, pred_mc, metric="DSC", multi_class=True,
                     n_classes=5)   
  # returns array of np.float64 e.g. [0.9, 0.2, 0.6, 0.0, 0.4]
  # for each class, one score