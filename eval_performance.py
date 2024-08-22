# This is demanding. Run on Colab
#%%
from torch_fidelity import calculate_metrics
import os
import numpy as np

#%%
current_directory = os.getcwd() #parameters
original_image_path = os.path.join(current_directory,"images_sample\original")
gen_image_path=os.path.join(current_directory,"images_sample\generated")

#%%
# details of the function: https://torch-fidelity.readthedocs.io/en/latest/usage_api.html
metrics = calculate_metrics(input1=original_image_path, 
                            input2=gen_image_path, 
                            cuda=False,
                            isc=False,
                            fid=True,
                            kid=False,
                            verbose=True,                            
                            )

print(metrics)
# %%
