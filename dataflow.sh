#!/bin/bash

gcloud pubsub subscriptions create mlb-data-subscription --topic=mlb-data-topic

JOB_NAME_GAMES="dodgers-mlb-data-pipeline-games"
JOB_NAME_PLAYS="dodgers-mlb-data-pipeline-plays"
REGION="us-central1"
JOB_NAME_PLAYER_STATS="dodgers-mlb-data-pipeline-player-stats"


gcloud dataflow jobs run $JOB_NAME_GAMES \
  --gcs-location=gs://dataflow-templates-$REGION/latest/PubSub_Subscription_to_BigQuery \
  --region=$REGION \
  --parameters=inputSubscription=projects/gem-rush-007/subscriptions/mlb-data-subscription,outputTableSpec=gem-rush-007:dodgers_mlb_data_2024.games


gcloud dataflow jobs run $JOB_NAME_PLAYS \
  --gcs-location=gs://dataflow-templates-<REGION>/latest/PubSub_Subscription_to_BigQuery \
  --region=$REGION \
  --parameters=inputSubscription=projects/gem-rush-007/subscriptions/mlb-data-subscription,outputTableSpec=gem-rush-007:dodgers_mlb_data_2024.plays


gcloud dataflow jobs run $JOB_NAME_PLAYER_STATS \
  --gcs-location=gs://dataflow-templates-$REGION/latest/PubSub_Subscription_to_BigQuery \
  --region=$REGION \
  --parameters=inputSubscription=projects/gem-rush-007/subscriptions/mlb-data-subscription,outputTableSpec=gem-rush-007:dodgers_mlb_data_2024.player_stats