from pathlib import Path

parent = Path(__file__).parent
input_file = Path(__file__).parents[2] / "tests/test.a3m"

AF_CLUSTER_EXEC = parent.parents[2] / "AF_Cluster/scripts/ClusterMSA.py"

cmd = f"python {AF_CLUSTER_EXEC} -i {input_file} -o {parent}/original_fixed_eps --eps_val 8 --min_samples 10 --run_PCA --run_TSNE cluster"

import subprocess

out = subprocess.run(cmd, shell=True, check=True)
