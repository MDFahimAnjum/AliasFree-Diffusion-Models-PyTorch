# This is demanding. Run on Colab
#%%
from torch_fidelity import calculate_metrics
import os
import numpy as np

#%%
current_directory = os.getcwd() #parameters
original_image_path = os.path.join(current_directory,"images_sample\original")
gen_image_path=os.path.join(current_directory,"images_sample\generated")
gen_image_path2=os.path.join(current_directory,"images_sample\generated_F")
control_path=os.path.join(current_directory,"images_sample\control")

#%%
# details of the function: https://torch-fidelity.readthedocs.io/en/latest/usage_api.html
metrics0 = calculate_metrics(input2=original_image_path, 
                            input1=gen_image_path, 
                            cuda=True,
                            isc=False,
                            fid=True,
                            kid=False,
                            verbose=True,                            
                            )

print(metrics0)
# %%
metrics1 = calculate_metrics(input2=original_image_path, 
                            input1=gen_image_path2, 
                            cuda=False,
                            isc=False,
                            fid=True,
                            kid=False,
                            verbose=True,                            
                            )

print(metrics1)

# %%
metrics2 = calculate_metrics(input2=original_image_path, 
                            input1=control_path, 
                            cuda=False,
                            isc=False,
                            fid=True,
                            kid=False,
                            verbose=True,                            
                            )

print(metrics2)