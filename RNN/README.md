# Pytorch C++ & RNN

# Installation

Download the zip from pytorch https://pytorch.org/get-started/locally/  
Use the C++/Java and the CPU version since are task are light so no gpu needed for now
Unzip the downloaded zip file into a directory of your choice

# Link folders

ln -s ../Results/ .
ln -s ../Data .

# Results

Epochs = 300
Overall Metrics:
  RMSE: 0.014321 rad = 0.821 deg
  MAE:  0.010296 rad = 0.590 deg

Training Complete
Elapsed(s) = 222
Evaluating RMSE for each prediction step
Calculated all predictions size: [3383, 5, 3]
Overall Metrics:
  RMSE (all): 0.012067 rad = 0.691 deg
  MAE  (all): 0.008878 rad = 0.509 deg

RMSE per angle (all samples, all steps):
  Roll  RMSE: 0.015381 rad = 0.881 deg
  Pitch RMSE: 0.011396 rad = 0.653 deg
  Yaw   RMSE: 0.008390 rad = 0.481 deg

RMSE per step
Step | Roll (deg) | Pitch (deg) | Yaw (deg)
-----+------------+-------------+-----------
   1 |   0.529896 |    0.475151 |  0.277213
   2 |   0.701064 |    0.426487 |  0.336128
   3 |   0.843295 |    0.504746 |  0.459770
   4 |   1.009834 |    0.748155 |  0.589079
   5 |   1.174697 |    0.953750 |  0.638116
