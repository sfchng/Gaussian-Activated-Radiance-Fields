#from google.cloud import storage
import logging
import os
import glob

def upload_folders(bucket_name, source_folder, destination_blob_name):

    assert os.path.isdir(source_folder)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)   

    for local_file in glob.glob(source_folder + '/**'):
        remote_path = f'{destination_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
        else:
            src_folder = "{}/{}".format(source_folder, os.path.basename(local_file))

            upload_folders(bucket_name, src_folder, destination_blob_name)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    ## Initialise a client ##
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def download_folders(bucket_name, blob_name):
    """Download a file from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # dest_file_name = "local/path/to/file"
    # The ID of your GCS object
    # blob_name = "storage-object-name"

    ## Initialise a client ##
    storage_client = storage.Client()

    ## Create a bucket object for our bucket ##
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=blob_name)
    print("Downloading blob")
    for blob in blobs:
        print("Blobs: {}".format(blob.name))
        destination = os.path.dirname(blob.name)
        os.makedirs(destination, exist_ok=True)
        blob.download_to_filename(blob.name)
        print("Exported {} to {}".format(blob.name, destination))



def download_blob(bucket_name, source_file_name, blob_name):
    """Uploads a file from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # blob_name = "storage-object-name"

    ## Initialise a client ##
    storage_client = storage.Client()

    ## Create a bucket object for our bucket ##
    bucket = storage_client.bucket(bucket_name)

    ## Create a blob object from the filepath ##
    blob = bucket.blob(blob_name)

    ## Download the file to a destination ##
    blob.download_to_filename(source_file_name)

    print(
        "File {} downloaded to {}.".format(
            blob, source_file_name
        )
    )