# Data Directory

This folder stores the synthetic salary dataset used to train and demo the application.

## Generated Dataset

The training script automatically generates `processed/salary_data.csv` the first time it runs. The generator crafts realistic salary values using:

- Years of experience and company tenure
- Job title and education level
- Employment type (full-time, contract, intern)
- Remote work ratio and country
- Company size and age demographics

Re-run `python train.py --refresh-data` to regenerate the dataset with the same random seed for reproducibility. The raw synthetic data is reproducible because the generator seeds NumPy's random number generator.

