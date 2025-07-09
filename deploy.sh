#!/bin/bash
docker build -t hybrid-field-mapper .
heroku container:push web -a hybrid-field-mapper
heroku container:release web -a hybrid-field-mapper
heroku config:set DATABASE_URL=postgresql://user:pass@host/db \
                 SECRET_KEY=your-secret-key \
                 IBMQ_TOKEN=your-ibmq-token \
                 SLACK_TOKEN=your-slack-token \
                 SENTRY_DSN=your-sentry-dsn \
                 METRICS_TOKEN=your-metrics-token \
                 REDIS_URL=redis://localhost:6379 \
                 CELERY_BROKER_URL=redis://localhost:6379 \
                 CELERY_RESULT_BACKEND=redis://localhost:6379 -a hybrid-field-mapper
