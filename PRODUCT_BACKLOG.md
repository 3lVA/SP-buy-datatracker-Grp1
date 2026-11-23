# SP-buy Fraud Detection Product Backlog

Story points are proposed estimates. Assign owners during sprint planning.

## Group backlog

### PBI-01 Explore and prepare historical data

**User story** — As an AIML data engineer, I want to identify and fix data quality issues so that the team can train models on reliable inputs.

**Points** 5 · **Assigned to** —

- [ ] A Jupyter notebook explores the base customer, order, and label CSVs and documents data quality issues, distributions, outliers, and class imbalance.
- [ ] Reusable Python code applies the documented cleaning and transformations and produces a labelled training dataset.

### PBI-02 Deploy shared MLflow tracking

**User story** — As an AIML team member, I want shared experiment tracking so that I can compare runs across the team.

**Points** 5 · **Assigned to** —

- [ ] Dockerized MLflow runs on AWS Elastic Beanstalk.
- [ ] A team member can log a run and retrieve its parameters, metrics, and model artifact.

### PBI-03 Train and evaluate fraud models

**User story** — As a Trust & Safety analyst, I want to identify potentially fraudulent orders so that I can prioritize reviews before shipment.

**Points** 8 · **Assigned to** —

- [ ] Each team member trains at least two model versions from customer and order information and logs parameters, metrics, and model artifacts to MLflow.
- [ ] The team uses a consistent evaluation approach and documents how previous results informed feature engineering or training changes.
- [ ] Training features are checked for availability at review time to avoid leaking later outcomes.

### PBI-04 Register and promote the champion model

**User story** — As an AIML team member, I want a repeatable model review so that the team knows which model is approved for production use.

**Points** 3 · **Assigned to** —

- [ ] At the graded Scrum meeting, compare each member's first model using agreed metrics, register the top two in MLflow, and promote the top model as champion.
- [ ] Repeat the review when a new model version is trained.

### PBI-05 Load cleaned batches into AWS RDS

**User story** — As an IT infrastructure manager, I want incoming data loaded consistently into MySQL so that downstream workflows can use current records.

**Points** 5 · **Assigned to** —

- [ ] Python scripts accept public URLs for the matching customers_XXX.csv, orders_XXX.csv, and labels_XXX.csv batch files.
- [ ] The scripts reuse the historical cleaning and transformation rules and write queryable records to AWS RDS MySQL.
- [ ] RDS hostname, port, credentials, and the MLflow URL are supplied through environment variables or configuration files.

### PBI-06 Retrain and log models for each labelled batch

**User story** — As an AIML team member, I want an automated retraining workflow so that I can compare models after new labels arrive.

**Points** 8 · **Assigned to** —

- [ ] Once supplied with the three URLs, the pipeline loads the batch and runs every team member's contributed training workflow against historical and new data.
- [ ] All runs are logged to shared MLflow with names that identify the batch or pipeline run.
- [ ] The group pipeline finishes within three minutes without further manual intervention, and the resulting RDS records and MLflow runs are accessible to the team.

### PBI-07 Improve MLflow storage durability

**User story** — As an IT infrastructure manager, I want experiment records and model artifacts to survive container replacement so that the team can retain its experiment history.

**Points** 5 · **Assigned to** —

- [ ] The team documents SQLite and container-local storage limitations and completes the brief's web-based GenAI discussion in fewer than 10 prompting rounds.
- [ ] The team implements changes using AWS Academy Learner Lab services and summarizes which services changed.
- [ ] A test run and its artifact remain accessible after redeployment.

### PBI-11 Verify the workflow in the live interview

**User story** — As a project stakeholder, I want to see the team process an unseen batch live so that I can verify the workflow works with new data.

**Points** 3 · **Assigned to** —

- [ ] Rehearse with separate practice data. At the interview, use the tutor-provided URLs received during the slot and verify new RDS records, fresh MLflow runs, and updated individual analysis results.
- [ ] Keep the presentation within 10 minutes and cover data findings, feature engineering, live MLflow comparison, RDS and service design, pipeline automation, and team contributions.

## Individual backlog

Each student owns their own copy of the required items and independently chooses and implements their analysis.

### PBI-08 Analyze fraud patterns from current data

**User story** — As a Trust & Safety analyst, I want to see how fraud varies across customer or order groups so that I can focus my investigations.

**Points** 5 per student · **Assigned to** —

- [ ] Write three meaningful analytical queries against the latest data in the team's RDS database.
- [ ] Use pandas, matplotlib, and seaborn to present and interpret results for Trust & Safety.

### PBI-09 Refresh analytics with a local Prefect flow

**User story** — As a Trust & Safety analyst, I want analysis refreshed on every flow run so that I can use the latest data.

**Points** 5 per student · **Assigned to** —

- [ ] A local Prefect service runs a flow that reads current RDS data, reruns all three analyses, and refreshes the visualizations.
- [ ] Query results appear at minimum in the flow logs.
- [ ] Each student demonstrates the flow on their own laptop within three minutes.

### PBI-10 Deploy Prefect on AWS (optional challenge)

**User story** — As an IT infrastructure manager, I want the analytics workflow to run on AWS so that it can operate independently of a student's laptop.

**Points** 5 per student · **Assigned to** —

- [ ] A Dockerfile packages the Prefect web server, which runs on Elastic Beanstalk with an RDS backend.
- [ ] A separate service connects to the server and successfully runs the analytics flow.
