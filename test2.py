from utils.ML_support import print_model_prop
import numpy as np

arr = np.array([1,2,np.inf,3,4])
print(np.where(np.isinf(arr))[0])

# print_model_prop("/home/zvladimi/MLOIS/xgboost_results/l0063n0256s190to166/base_l0063n0256s190to166nu0-10wght4.304_0.17/model_info.pickle")

# print_model_prop("/home/zvladimi/MLOIS/xgboost_results/l0063n0256s190to166/base_l0063n0256s190to166nu0-10wght1.558_0.012/model_info.pickle")
