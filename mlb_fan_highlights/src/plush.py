import google.auth

credentials, project_id = google.auth.default()

if credentials and hasattr(credentials, 'service_account_email'):
    print(f"Service Account Email: {credentials.service_account_email}")
else:
    print("Could not determine service account email.")

