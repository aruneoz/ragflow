import time
import uuid

from google.cloud import monitoring_v3

def getMonitoringClient(projectId):
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{projectId}"

    return client, project_name

def writeMetric():
    client , project_name = getMonitoringClient("greenfielddemos")
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/file_counter"
    series.resource.type = "gke_container"
    series.resource.labels["instance_id"] = "svp-simulation-1"
    series.resource.labels["zone"] = "us-central1-c"
    series.resource.labels["cluster_name"] = "svp-simulation-1"
    series.resource.labels["container_name"] = "rag-worker-dataextraction"
    series.resource.labels["namespace_id"] = "default"
    series.resource.labels["pod_id"] = "rag-worker-dataextraction-6cbc57698c-2plpm"
    series.metric.labels["Description"] = "Total no of Files process during RAG pipeline"
    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10 ** 9)
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": seconds, "nanos": nanos}}
    )
    point = monitoring_v3.Point({"interval": interval, "value": {"double_value": 1}})
    series.points = [point]
    client.create_time_series(name=project_name, time_series=[series])


writeMetric()

