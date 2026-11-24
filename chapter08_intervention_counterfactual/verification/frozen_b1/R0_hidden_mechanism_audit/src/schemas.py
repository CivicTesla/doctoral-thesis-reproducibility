O0_FIELDS = (
    "local_queue_length", "head_of_line_wait", "local_request_failures",
    "locally_observed_busy_flag", "time_since_last_success",
    "local_retry_count", "recent_local_throughput",
)
O1_FIELDS = (
    "aggregate_mean_queue", "aggregate_busy_fraction",
    "aggregate_failure_rate", "number_of_active_requesters",
    "simple_contention_graph_degree",
)
O2_FIELDS = (
    "resource_state_version", "last_release_version", "observed_holder_id",
    "holder_confirmed_active", "request_timestamp", "accept_timestamp",
    "release_timestamp", "time_since_last_release",
    "number_of_conflicting_versions", "number_of_nodes_reporting_busy",
    "number_of_nodes_reporting_free", "resource_probe_response",
)
FORBIDDEN_FIELDS = ("mechanism", "true_mechanism", "label", "config_filename")

FIELD_BITS = {
    "resource_state_version": 16, "last_release_version": 16,
    "observed_holder_id": 8, "holder_confirmed_active": 1,
    "request_timestamp": 16, "accept_timestamp": 16,
    "release_timestamp": 16, "time_since_last_release": 16,
    "number_of_conflicting_versions": 8, "number_of_nodes_reporting_busy": 8,
    "number_of_nodes_reporting_free": 8, "resource_probe_response": 8,
}

def evidence_bytes(fields):
    """Packed payload size, rounded up once for the complete subset."""
    unknown = set(fields) - FIELD_BITS.keys()
    if unknown:
        raise KeyError(f"unknown evidence fields: {sorted(unknown)}")
    return (sum(FIELD_BITS[name] for name in fields) + 7) // 8
