import json
import time

from trainscope.buffer import Flusher, RingBuffer


class FlakySink:
    def __init__(self):
        self.up = True
        self.received = []

    def send(self, batch):
        if not self.up:
            raise ConnectionError("down")
        self.received.append(batch)


def test_ring_buffer_drops_oldest():
    b = RingBuffer(maxlen=3)
    for i in range(5):
        b.add_metric({"i": i})
    assert [m["i"] for m in b.metrics] == [2, 3, 4]
    assert b.dropped == 2


def test_drain_respects_batch_size():
    b = RingBuffer()
    for i in range(10):
        b.add_metric({"i": i})
    out = b.drain(4)
    assert len(out["metrics"]) == 4 and len(b.metrics) == 6


def test_spools_when_sink_down_and_replays(tmp_path):
    sink = FlakySink()
    b = RingBuffer()
    f = Flusher(b, sink, tmp_path / "s.jsonl", interval=60, retry_interval=0)
    sink.up = False
    for i in range(5):
        b.add_metric({"i": i})
    f.flush_now()
    assert f.spooled and not sink.received
    assert json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])["metrics"][0]["i"] == 0
    sink.up = True
    f.flush_now(final=True)
    assert not f.spooled
    assert sum(len(x["metrics"]) for x in sink.received) == 5


def test_flush_thread_never_raises(tmp_path):
    class Boom:
        def send(self, batch):
            raise RuntimeError("boom")

    b = RingBuffer()
    f = Flusher(b, Boom(), tmp_path / "s.jsonl", interval=0.01)
    f.start()
    b.add_metric({"i": 1})
    time.sleep(0.1)
    assert f.is_alive()
    f.stop()
