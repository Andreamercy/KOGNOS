"""
tests/unit/test_synthetic_gen.py

Unit tests for the synthetic cluster failure data generator.

Verifies that generated telemetry windows have the correct structure,
that all scenario types produce valid data, and that labels are bounded.
"""

import pytest
import time

from graph.data.synthetic_gen import (
    generate_window,
    generate_labelled_window,
    generate_dataset,
    FailureScenario,
    LabelledWindow,
)
from graph.builder.cluster_graph import TelemetryWindow, PodNode, FlowEdge


class TestGenerateWindow:
    """Tests for the basic generate_window function."""

    def test_returns_telemetry_window(self):
        window = generate_window()
        assert isinstance(window, TelemetryWindow)

    def test_has_pods(self):
        window = generate_window()
        assert len(window.pods) > 0

    def test_pods_are_pod_nodes(self):
        window = generate_window()
        for pod in window.pods:
            assert isinstance(pod, PodNode)

    def test_flows_are_flow_edges(self):
        window = generate_window()
        for flow in window.flows:
            assert isinstance(flow, FlowEdge)

    def test_timestamps_are_ordered(self):
        window = generate_window()
        assert window.window_start_ns < window.window_end_ns

    def test_timestamps_are_recent(self):
        before = time.time_ns()
        window = generate_window()
        after  = time.time_ns()
        assert window.window_end_ns <= after
        assert window.window_start_ns >= before - 35_000_000_000  # 35s window

    def test_deterministic_with_seed(self):
        w1 = generate_window(seed=42)
        w2 = generate_window(seed=42)
        assert [p.name for p in w1.pods] == [p.name for p in w2.pods]
        assert [p.cpu_usage for p in w1.pods] == [p.cpu_usage for p in w2.pods]

    def test_different_seeds_differ(self):
        w1 = generate_window(seed=1)
        w2 = generate_window(seed=2)
        # Very unlikely to be identical
        cpus1 = [p.cpu_usage for p in w1.pods]
        cpus2 = [p.cpu_usage for p in w2.pods]
        assert cpus1 != cpus2


class TestGenerateLabelledWindow:
    """Tests for the labelled window generator used in training."""

    def test_returns_labelled_window(self):
        lw = generate_labelled_window()
        assert isinstance(lw, LabelledWindow)

    def test_labels_length_matches_pods(self):
        lw = generate_labelled_window()
        assert len(lw.labels) == len(lw.window.pods)

    def test_labels_bounded(self):
        for _ in range(10):
            lw = generate_labelled_window()
            assert all(0.0 <= lbl <= 1.0 for lbl in lw.labels), \
                f"Label out of bounds: {lw.labels}"

    @pytest.mark.parametrize("scenario", list(FailureScenario))
    def test_all_scenarios_generate(self, scenario):
        lw = generate_labelled_window(scenario=scenario, seed=0)
        assert lw.scenario == scenario
        assert len(lw.window.pods) > 0

    def test_oomkill_has_high_label(self):
        lw = generate_labelled_window(scenario=FailureScenario.OOMKILL, seed=0)
        max_label = max(lw.labels)
        assert max_label > 0.5, f"OOMKill scenario should produce high labels, got {max_label}"

    def test_normal_has_lower_labels(self):
        lw = generate_labelled_window(scenario=FailureScenario.NORMAL, seed=0)
        max_label = max(lw.labels)
        assert max_label < 0.7, f"Normal scenario should produce low labels, got {max_label}"

    def test_affected_pods_listed(self):
        lw = generate_labelled_window(scenario=FailureScenario.OOMKILL, seed=0)
        if lw.affected:
            # Affected pods should have labels above threshold
            pod_names = [p.name for p in lw.window.pods]
            for affected_pod in lw.affected:
                assert affected_pod in pod_names


class TestGenerateDataset:
    """Tests for the bulk dataset generator used in training."""

    def test_returns_list(self):
        dataset = generate_dataset(n_windows=5, seed=42)
        assert isinstance(dataset, list)

    def test_correct_length(self):
        dataset = generate_dataset(n_windows=10, seed=42)
        assert len(dataset) == 10

    def test_mixed_scenarios(self):
        dataset  = generate_dataset(n_windows=100, seed=42)
        scenarios = {lw.scenario for lw in dataset}
        # With 100 samples and weighted sampling, expect > 1 scenario
        assert len(scenarios) > 1, f"Expected multiple scenarios, got {scenarios}"


class TestPodNodeFeatures:
    """Verify pod node feature invariants."""

    def test_cpu_usage_bounded(self):
        for scenario in FailureScenario:
            window = generate_window(scenario=scenario, seed=0)
            for pod in window.pods:
                assert 0.0 <= pod.cpu_usage <= 1.0, \
                    f"cpu_usage out of range for {pod.name}: {pod.cpu_usage}"

    def test_mem_usage_bounded(self):
        for scenario in FailureScenario:
            window = generate_window(scenario=scenario, seed=0)
            for pod in window.pods:
                assert 0.0 <= pod.mem_usage <= 1.0, \
                    f"mem_usage out of range for {pod.name}: {pod.mem_usage}"

    def test_error_rate_bounded(self):
        for scenario in FailureScenario:
            window = generate_window(scenario=scenario, seed=0)
            for pod in window.pods:
                assert 0.0 <= pod.error_rate <= 1.0, \
                    f"error_rate out of range: {pod.error_rate}"

    def test_restart_count_nonnegative(self):
        window = generate_window(seed=0)
        for pod in window.pods:
            assert pod.restart_count >= 0


class TestFlowEdgeFeatures:
    """Verify flow edge feature invariants."""

    def test_flow_pods_exist_in_window(self):
        window    = generate_window(seed=0)
        pod_names = {p.name for p in window.pods}
        for flow in window.flows:
            assert flow.src_pod in pod_names, \
                f"src_pod {flow.src_pod} not in pod list"
            assert flow.dst_pod in pod_names, \
                f"dst_pod {flow.dst_pod} not in pod list"

    def test_bytes_nonnegative(self):
        window = generate_window(seed=0)
        for flow in window.flows:
            assert flow.bytes_per_sec >= 0.0
