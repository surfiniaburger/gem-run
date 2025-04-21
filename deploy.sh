#!/bin/bash

git pull

GCP_REGION='us-central1'
GCP_PROJECT='silver-455021'
AR_REPO='gem-run'
SERVICE_NAME='mlb'

gcloud builds submit \
    --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
    --port=8080 \
    --image "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
    --region="$GCP_REGION" \
    --platform=managed \
    --project="$GCP_PROJECT" \
    --service-account=mlb-agent-runner@silver-455021.iam.gserviceaccount.com \
    --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION"