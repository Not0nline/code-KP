import os
from prometheus_api_client import PrometheusConnect

prometheus_server = os.getenv('PROMETHEUS_URL', 'http://prometheus-operated.monitoring.svc.cluster.local:9090')
prom = PrometheusConnect(url=prometheus_server, disable_ssl=True)
traffic_query = 'sum(rate(load_tester_requests_total{service="Combined"}[1m]))'
traffic = prom.custom_query(query=traffic_query)
print(traffic)
