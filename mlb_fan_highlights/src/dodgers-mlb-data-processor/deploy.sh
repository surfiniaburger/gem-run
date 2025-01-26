#!/bin/bash

git pull

GCP_REGION='us-central1'
GCP_PROJECT='gem-rush-007'
AR_REPO='dodgers-007'
SERVICE_NAME='dodgers-mlb'

gcloud builds submit \
    --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
    --port=8080 \
    --image "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
    --region="$GCP_REGION" \
    --platform=managed \
    --project="$GCP_PROJECT" \
    --service-account=cloud-run-invoker@gem-rush-007.iam.gserviceaccount.com \
    --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION"