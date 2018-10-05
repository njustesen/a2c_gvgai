from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os

model_dir = "./results/solarfox-ls-pcg-random-10/models/8a270ecc-adf4-11e8-b25d-000d3a60d0e1/"
checkpoint_path = os.path.join(model_dir, "model-22000000")

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)
