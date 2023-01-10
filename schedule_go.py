import os

if __name__ == "__main__":
    for seed in range(100):
        for n in [10, 25, 50, 100, 250]:
            command = f"sbatch slurm.sh gcn-link-prediction-go.py -seed {seed} -n {n}"

            os.system(command)
