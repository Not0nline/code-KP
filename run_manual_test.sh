#!/bin/bash
sudo kubectl run my-manual-test \
    --image=4dri41/stress-test:latest \
    --image-pull-policy=Always \
    --restart=Never \
    --env="TARGET=combined" \
    --env="SCENARIO=low" \
    --env="DURATION=300" \
    --env="TEST_NAME=my-manual-test" \
    --env="TEST_PREDICTIVE=true" \
    --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600"
