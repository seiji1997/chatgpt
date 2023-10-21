import os
from google.cloud import bigquery

def migrate_to_bigquery(data, context):
    project_id = os.environ['GCP_PROJECT']
    dataset_id = 'your_dataset_id'
    table_id = 'titanic_data'

    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        source_format=bigquery.SourceFormat.CSV,
    )

    load_job = client.load_table_from_uri(data, table_ref, job_config=job_config)
    load_job.result()

    print(f'File {data} loaded to {table_ref.path}')

# The function triggers when a new file is created in the GCP Cloud Storage bucket.
