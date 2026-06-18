from pathlib import Path

from google.cloud import storage
from google.auth.credentials import Credentials
from google.oauth2 import service_account


SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def credentials_from_info(service_account_info: dict | None = None) -> Credentials | None:
    if service_account_info:
        return service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES,
        )
    return None


def storage_client_from_info(service_account_info: dict | None = None) -> storage.Client:
    credentials = credentials_from_info(service_account_info)
    if credentials:
        return storage.Client(credentials=credentials, project=service_account_info.get("project_id"))
    return storage.Client()


def download_file(bucket_name: str, source_blob: str, destination: str | Path, client: storage.Client | None = None) -> Path:
    client = client or storage_client_from_info()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    bucket.blob(source_blob).download_to_filename(str(destination))
    return destination
