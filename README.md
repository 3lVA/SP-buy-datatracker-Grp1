# SP-buy-datatracker-Grp1

## Deploy Prefect to AWS

The `spbuy-prefect` server hosts the team dashboard and API. It restores Prefect's SQLite history from private S3 on startup and checkpoints changes to the same object. The ETL runner loads CSV batches into the existing RDS MySQL database and sends ingestion and retraining results to MLflow.

1. Use a private S3 bucket with public access blocked. Upload the verified Prefect database backup to a new key such as `prefect-history/group1/prefect.db` and keep a separate original backup. Keep both outside MLflow's artifact and backup prefixes. If there is already a live Prefect checkpoint, back it up and compare the run histories before replacing it. The Beanstalk instance profile needs `s3:GetObject` and `s3:PutObject` for the Prefect checkpoint key.
2. Use the existing Docker web server environment and keep one instance for the SQLite database. The included Docker Compose bundle runs a Caddy proxy alongside Prefect. Package the deployment files from the repository root in PowerShell.

   ```powershell
   $prefectZip = Join-Path $env:TEMP 'spbuy-prefect.zip'
   Compress-Archive -LiteralPath .\deploy_prefect_aws\Dockerfile, .\deploy_prefect_aws\persistent_server.py, .\deploy_prefect_aws\docker-compose.yml, .\deploy_prefect_aws\Caddyfile -DestinationPath $prefectZip -Force
   ```

3. For the included proxy, use the instance's Elastic IP or a team-controlled DNS name that points to it. The instance security group needs inbound TCP 443 for sign-in and ACME certificate validation. Caddy requests a short-lived certificate and redirects HTTP requests to HTTPS, except for `/api/health`. Verify that `/api/health` succeeds over HTTPS before asking teammates to sign in.
4. Set these Beanstalk environment properties, then upload the ZIP. Use a Secrets Manager or Parameter Store reference for the API credential.

   | Setting | Value |
   | --- | --- |
   | `PREFECT_SERVER_DATABASE_CONNECTION_URL` | `sqlite+aiosqlite:////tmp/prefect.db` |
   | `PREFECT_PERSISTENCE_BUCKET` | The private S3 bucket name |
   | `PREFECT_PERSISTENCE_PREFIX` | The key prefix before `/prefect.db`, for example `prefect-history/group1` |
   | `PREFECT_SERVER_API_AUTH_STRING` | The team's Prefect `username:password` |
   | `PREFECT_PUBLIC_HOST` | The Elastic IP or DNS name, with no scheme or path |
   | `PREFECT_UI_API_URL` | `https://` followed by the same public host and `/api` |

   Startup fails if the S3 checkpoint is missing, inaccessible, corrupt, or empty. The container checks for new database writes every 30 seconds and tries a final checkpoint on graceful shutdown. An abrupt loss can discard writes since the last successful checkpoint. Check the Beanstalk logs for the restored run count and a successful checkpoint after deployment.

5. Put the AWS runner in a network that can reach the existing MySQL database. Allow MySQL port `3306` from the runner's security group. Set `PREFECT_API_URL` to the same value as the server's `PREFECT_UI_API_URL`. Set `PREFECT_API_AUTH_STRING`, the existing `RDS_*` settings, and `MLFLOW_TRACKING_URI` in the service environment or its private `.env`. Set `RDS_SSL_CA` to the CA file path on the runner. Run these commands from the repository root and use the last command as the managed service that starts on boot.

   ```sh
   python -m pip install -r requirements.txt -r pipelines/requirements.txt "prefect==3.9.5"
   python load_rds.py --check
   python -m pipelines.etl_pipeline
   ```
6. When `automated-data-ingestion/data-ingestion` shows **Ready**, open **Run → Custom run** and provide `customers_url`, `orders_url`, and `labels_url`. Set `retrain=True` to ingest the batch and retrain from the updated MySQL data, or set `retrain=False` to ingest only. Set `MLFLOW_TRACKING_URI` to the remote AWS MLflow server on the runner so it can log both results there. Each batch creates an `ETL_Ingestion_` run with before and after table counts, affected rows, and skipped labels. The audit run and any model runs share the `etl_run_id` tag in MLflow. The CSV data itself is stored in MySQL. After a lab restart, check that Prefect history persists and the runner returns to **Ready**.

### Open the team dashboard

Open the shared HTTPS URL in Chrome and sign in. To open it from VS Code PowerShell, put the shared API address in `PREFECT_API_URL` in your local `.env`. `prefect deployment ls` also uses `PREFECT_API_AUTH_STRING` from that file. Run these commands from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$prefectApi = dotenv -f .env get PREFECT_API_URL
if ($LASTEXITCODE -eq 0 -and $prefectApi -match '^https://[^/]+/api/?$') {
    Start-Process ($prefectApi -replace '/api/?$', '/')
}
prefect deployment ls
```
