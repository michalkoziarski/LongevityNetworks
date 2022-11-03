import os

if __name__ == "__main__":
    for seed in range(100):
        command = f"sbatch slurm.sh gcn-link-prediction-multiseed.py -seed {seed}"

        os.system(command)
