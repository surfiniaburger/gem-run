#!/bin/bash

gcloud compute networks create my-custom-network --subnet-mode=custom

gcloud compute networks subnets create us-central1-subnet \
    --network=my-custom-network \
    --region=us-central1 \
    --range=10.100.0.0/24

gcloud compute networks subnets create us-east1-subnet \
    --network=my-custom-network \
    --region=us-east1 \
    --range=10.200.0.0/24

