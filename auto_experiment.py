import os
import subprocess
print("CPU Count is :", os.cpu_count())

paths = [
   # 'path/to/trained_fsl_model.pth', 
   # 'path/to/trained_fsl_model.pth'
]

# results will be saved as a txt file in "./outputs" folder. The `--use_euclidean` flag is for euclidean-distance-based classifier.
fsl_testing = [
   f"python test_fsl.py --model_class FEAT --backbone_class ConvNet --dataset animals --model_path {paths[0]} --spt_expansion 3 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 5000 --note your_experiment_note",
   f"python test_fsl.py --model_class DeepSet --backbone_class ConvNet --dataset animals --use_euclidean --model_path {paths[1]} --spt_expansion 3 --qry_expansion 3 --aug_type image_translator --num_eval_episodes 5000 --note your_experiment_note",
]

commands = fsl_testing
for command in commands:
   print(f"Command: {command}")
   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
