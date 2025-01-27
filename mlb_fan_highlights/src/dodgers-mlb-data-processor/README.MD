## Before deploying add this command to deploy.sh, before the command to build the docker image

```bash
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
```