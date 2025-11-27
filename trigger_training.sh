#!/bin/bash
python -c "import requests; print(requests.post('http://localhost:5000/api/models/train_all', json={}).text)"
