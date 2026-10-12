# Platform Runbook

This runbook describes how the Orders platform is built, released, and operated. It is a fictional practice document for the RAG course.

## Service Layout

The platform has two kinds of workloads. Long-running HTTP APIs are Node.js services packaged as containers and run on AWS Fargate behind an Application Load Balancer. Short, event-driven jobs run as AWS Lambda functions triggered by SQS queues and EventBridge schedules.

Each service owns its own repository, its own pipeline, and its own IAM task role. Services never share a task role.

## Release Process

### Fargate Services

Fargate services are released with blue-green deployments. The pipeline starts a new task set (green) alongside the running one (blue) and waits for load balancer health checks to pass on the green tasks.

Traffic then shifts in two steps: 10 percent of requests go to green for 10 minutes, and if the error rate stays under 1 percent the remaining 90 percent follows. The blue task set is kept for 60 minutes after the shift so a rollback is immediate.

### Lambda Functions

Lambda functions are released by publishing a new version and moving the `live` alias to it. Alias traffic shifting sends 10 percent of invocations to the new version for 5 minutes before moving the rest.

### Rollback

To roll back a Fargate service, re-route the load balancer listener to the blue target group. This takes effect in under a minute and does not require a rebuild. After the 60 minute window the blue tasks are gone, and a rollback means redeploying the previous image tag.

To roll back a Lambda function, point the `live` alias at the previous version number.

## Configuration and Secrets

Non-secret configuration lives in AWS Systems Manager Parameter Store under the path `/orders/<service>/<environment>/`. Secrets such as API keys live in AWS Secrets Manager and are rotated every 90 days.

Services read secrets at startup through their task role. Secrets are never written to environment files, container images, or logs.

## Timeouts and Retries

Every outbound HTTP call sets an explicit timeout of 3 seconds and retries at most twice with exponential backoff and jitter. Lambda functions that read from SQS use a visibility timeout of six times the function timeout, and messages that fail three times move to a dead-letter queue.

Fargate services must respond to the load balancer health check within 2 seconds. A task that fails three consecutive health checks is replaced.

## Observability

Services write structured JSON logs to CloudWatch Logs with a correlation id on every line. Each service publishes request latency, error rate, and saturation metrics.

Alerts fire when the 5 minute error rate exceeds 2 percent or when p99 latency exceeds 1.5 seconds for 10 minutes. Dead-letter queue depth above zero raises a low-priority ticket rather than a page.

## Incident Response

The on-call engineer acknowledges a page within 10 minutes. The first action is always to check whether a release happened in the last hour, and to roll it back if so, before investigating further.

Incidents are graded by customer impact. Severity 1 means checkout is unavailable and requires an incident commander and updates every 30 minutes. Severity 2 means degraded performance and requires updates every 2 hours.

A written review is due within 5 business days of any Severity 1 incident. Reviews are blameless and focus on what made the failure possible.
