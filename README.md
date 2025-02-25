[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# "Adaptive Bounded Exploration and Intermediate Actions for Data Debiasing"(https://doi.org/10.1287/ijoc.2024.0651) 

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [Creative Commons Attribution-NonCommercial-NoDerivs](LICENSE).

The data in this repository is a snapshot of the data that was used in the research reported on in the paper
[Adaptive Bounded Exploration and Intermediate Actions for Data Debiasing](https://doi.org/10.1287/ijoc.2024.0651) by Yifan Yang, Yang Liu, and Parinaz Naghizadeh.

## Cite 
To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0651

https://doi.org/10.1287/ijoc.2024.0651.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{Yang2025,
  author =        {Yifan Yang and Yang Liu and Parinaz Naghizadeh},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Adaptive Bounded Exploration and Intermediate Actions for Data Debiasing}},
  year =          {2025},
  doi =           {10.1287/ijoc.2024.0651.cd},
  url =           {https://github.com/INFORMSJoC/2024.0651},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0651},
}  
```
### Required packages/modules

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.utils import shuffle  
from sklearn import metrics  
from sklearn.metrics import confusion_matrix  
import sklearn.preprocessing as preprocessing  

from scipy.stats import norm, beta  
from statistics import median  
from random import choices  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

from responsibly.dataset import build_FICO_dataset  
from responsibly.fairness.interventions.threshold import find_thresholds  
 

### Assumptions  

1: Individuals have single dimensional feature or score $x \in \mathcal{R}$  
2: Individuals belong to one of two groups: $G_a, G_b$  
3: Individuals are either qualified or unqualified, and have a true label $y \in$  {0,1}  
4: A threshold-based classifier $\theta_g$ is used to make 'accept/reject' decision on these individuals  
5: Biased estimated feature distribution $\hat{f}^y_g(x)$ and true distribution $f^y_g(x)$ are differ in single parameter  
6: Measure of bias: differences between two parameters $|\hat{\omega}^y_g - \omega^y_g|$

### Experiments : baseline models  

Model 1: Exploititaiton only model (Collect all data above threshold)  
- See file [Exploitation_Only_Baseline.ipynb](Exploitation_Only_Baseline.ipynb) 

Model 2: Pure exploration (Collect all data without considering risk)  
- See file [Pure_Exploration_Baseline.ipynb](Pure_Exploration_Baseline.ipynb)

### Experiments: symmetric and asymmetric distributions

Synthetic symmetric Data: see file [Gaussian_Experiment.ipynb](Gaussian_Experiment.ipynb)  
Synthetic asymmetric Data: see file [Beta_Experiment.ipynb](Beta_Experiment.ipynb)  

### Experiments:

Real Dataset (**Adult, FICO, Fair**): see file [Adult_FICO_Gaussian_Fair.ipynb](Adult_FICO_Gaussian_Fair.ipynb)  
Retiring Adult Dataset: see file [Retiring Adult.ipynb](Retiring_adult_adaptive_debiasing.ipynb)  
Two Stage MDP Experiments: see file [MDP.ipynb](MDP.ipynb)  


