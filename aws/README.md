# AWS Deployment Guide

This folder contains an AWS Serverless Application Model (SAM) template and Lambda handler for deploying the trained salary prediction model as a managed inference API.

## Overview

- `template.yaml` provisions two Lambda functions behind API Gateway (POST for scoring, GET for health checks) and a Lambda layer to ship the serialized model artifact.
- `lambda_handler.py` loads the pre-trained Scikit-learn pipeline once per container, then scores incoming requests.
- `requirements.txt` lists the runtime dependencies that must be packaged with the deployment.

## Prerequisites

- AWS CLI v2 and SAM CLI installed and configured with deployment credentials.
- A trained model artifact at `models/salary_model.joblib` (run `python train.py` if missing).
- Python 3.11 for building the deployment bundle.

## Deployment Steps

1. **Install dependencies for the Lambda bundle**

   ```bash
   cd /Users/sverma/Github/MyProject
   python3 -m venv .aws-venv
   source .aws-venv/bin/activate
   pip install -r aws/requirements.txt
   deactivate
   ```

2. **Package the Lambda layer with the trained model**

   ```bash
   mkdir -p aws/.aws-sam/build/ModelLayer/models
   cp models/salary_model.joblib aws/.aws-sam/build/ModelLayer/models/
   ```

3. **Build the SAM application**

   ```bash
   sam build --template aws/template.yaml --use-container
   ```

4. **Deploy**

   ```bash
   sam deploy \
     --guided \
     --capabilities CAPABILITY_IAM \
     --stack-name salary-prediction-api
   ```

   During the guided deploy you can accept defaults, but ensure the Lambda layer is packaged and uploaded.

5. **Invoke the endpoint**

   After deployment, SAM prints an invoke URL. Send predictions with:

   ```bash
   curl -X POST "$ENDPOINT" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "experience_years": 5,
         "years_with_company": 2,
         "company_size": 1000,
         "age": 29,
         "education_level": "Bachelor's",
         "job_title": "Software Engineer",
         "country": "United States",
         "employment_type": "Full-time",
         "remote_ratio": "Hybrid"
       }
     }'
   ```

## Production Hardening Suggestions

- Store the trained model in Amazon S3 or SageMaker Model Registry and download it during deployment.
- Add request validation via API Gateway models and quotas to protect the endpoint.
- Enable X-Ray tracing and CloudWatch metrics/dashboards for observability.
- Swap SAM for AWS CDK if you prefer infrastructure as code in Python/TypeScript.

