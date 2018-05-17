from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os

model_dir = "./results/zelda-lvl-0/models/9aab32a2-5884-11e8-b99e-6c4008b68262/"
checkpoint_path = os.path.join(model_dir, "model-2000")

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)
