from google.cloud import storage


class GCS_Bucket(object):

    def __init__(self, bucket_name):
        client = storage.Client()
        self.bucket = client.get_bucket(bucket_name)

    def put(self, local_file_name, gcs_blob_name):
        blob = self.bucket.blob(gcs_blob_name)
        blob.upload_from_filename(filename=local_file_name)

    def take(self, gcs_blob_name, local_file_name):
        blob = self.bucket.get_blob(gcs_blob_name)
        if blob is None:
            raise ValueError("failed to locate the file {} in GCS".format(gcs_blob_name))
        blob.download_to_filename(filename=local_file_name)