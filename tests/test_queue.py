"""Tests for SQLite WAL Episode Queue and DLQ."""

from janus_graph.pipeline.queue import EpisodeQueue


def test_queue_enqueue_and_stats(temp_queue):
    stats = temp_queue.get_stats()
    assert stats.get("queued", 0) == 0
    assert stats.get("dlq", 0) == 0

    ep_id1 = temp_queue.enqueue("Episode content 1", group_id="test_group")
    ep_id2 = temp_queue.enqueue("Episode content 2", group_id="test_group")
    assert ep_id1 != ep_id2

    stats_after = temp_queue.get_stats()
    assert stats_after.get("queued") == 2
