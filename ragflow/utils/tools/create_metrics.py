import uuid

from google.api import label_pb2 as ga_label
from google.api import metric_pb2 as ga_metric
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/greenfielddemos"
descriptor = ga_metric.MetricDescriptor()
descriptor.type = "custom.googleapis.com/file_counter" + str(uuid.uuid4())
descriptor.metric_kind = ga_metric.MetricDescriptor.MetricKind.GAUGE
descriptor.value_type = ga_metric.MetricDescriptor.ValueType.DOUBLE
descriptor.description = "This is used to monitor the files processed during RAG processing flow ."

labels = ga_label.LabelDescriptor()
labels.key = "raglfow"
labels.value_type = ga_label.LabelDescriptor.ValueType.STRING
labels.description = "this metrics belongs to rag pipeline"
descriptor.labels.append(labels)

descriptor = client.create_metric_descriptor(
    name=project_name, metric_descriptor=descriptor
)
print("Created {}.".format(descriptor.name))