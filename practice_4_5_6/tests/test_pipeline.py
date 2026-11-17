import json
import pytest
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from pipeline import DocumentPipeline


@pytest.fixture(scope="module")
def setup_env():
    test_dir = Path(tempfile.mkdtemp())
    
    good_img = test_dir / "good.jpg"
    Image.new("RGB", (640, 640), color="white").save(good_img)
    
    corrupted_img = test_dir / "corrupted.jpg"
    corrupted_img.write_text("This is not an image binary")
    
    fake_img = test_dir / "fake.txt"
    fake_img.write_text("Hello")

    yield {"dir": test_dir, "good": good_img, "corrupt": corrupted_img, "txt": fake_img}
    shutil.rmtree(test_dir)


@pytest.fixture(scope="module")
def pipe():
    return DocumentPipeline("config.yaml")


def test_schema_keys(pipe, setup_env):
    res = pipe.run(str(setup_env["good"]))
    expected_keys = {"file", "status", "quality", "quality_confidence", 
                     "document_type", "document_type_confidence", 
                     "reason", "model_versions", "latency_ms", "timestamp"}
    assert set(res.keys()) == expected_keys
    assert isinstance(res["latency_ms"], int)
    assert isinstance(res["quality_confidence"], float)


@patch("pipeline.YOLO.__call__")
def test_good_image_accepted(mock_yolo, pipe, setup_env):
    mock_result = MagicMock()
    mock_result.probs.data = [0.05, 0.95]
    mock_yolo.return_value = [mock_result]
    
    res = pipe.run(str(setup_env["good"]))
    assert res["status"] == "accepted"
    assert res["quality"] == "good"
    assert res["document_type"] == "null"


@patch("pipeline.YOLO.__call__")
def test_bad_image_rejected(mock_yolo, pipe, setup_env):
    mock_result = MagicMock()
    mock_result.probs.data = [0.85, 0.15]
    mock_yolo.return_value = [mock_result]
    
    res = pipe.run(str(setup_env["good"]))
    assert res["status"] == "rejected"
    assert res["quality"] == "bad"
    assert res["reason"] == "poor_quality"


def test_corrupted_file(pipe, setup_env):
    res = pipe.run(str(setup_env["corrupt"]))
    assert res["status"] == "error"
    assert res["reason"] == "unsupported_file"


def test_non_image_file(pipe, setup_env):
    res = pipe.run(str(setup_env["txt"]))
    assert res["status"] == "error"
    assert res["reason"] == "unsupported_file"


def test_latency_budget(pipe, setup_env):
    latencies = []
    for _ in range(20):
        res = pipe.run(str(setup_env["good"]))
        latencies.append(res["latency_ms"])
    
    mean_latency = sum(latencies) / len(latencies)
    assert mean_latency < 500, f"Latency {mean_latency}ms exceeds 500ms budget"


@patch("pipeline.YOLO.__call__")
def test_threshold_config_shift(mock_yolo, setup_env):
    mock_result = MagicMock()
    mock_result.probs.data = [0.40, 0.60] 
    mock_yolo.return_value = [mock_result]

    pipe_strict = DocumentPipeline("config.yaml")
    pipe_strict.threshold = 0.25 
    res_strict = pipe_strict.run(str(setup_env["good"]))
    assert res_strict["status"] == "rejected"

    pipe_relaxed = DocumentPipeline("config.yaml")
    pipe_relaxed.threshold = 0.50
    res_relaxed = pipe_relaxed.run(str(setup_env["good"]))
    assert res_relaxed["status"] == "accepted"


@patch("pipeline.YOLO.__call__")
def test_batch_integration(mock_yolo, pipe, setup_env):
    mock_bad, mock_good = MagicMock(), MagicMock()
    mock_bad.probs.data = [0.99, 0.01]
    mock_good.probs.data = [0.01, 0.99]
    mock_yolo.side_effect = [[mock_good], [mock_bad]]

    test_files = [setup_env["good"], setup_env["good"], setup_env["corrupt"]]
    results = [pipe.run(str(f)) for f in test_files]

    out_json = setup_env["dir"] / "predictions.json"
    out_json.write_text(json.dumps(results, indent=2))
    
    assert out_json.exists()
    assert results[0]["status"] == "accepted"
    assert results[1]["status"] == "rejected"
    assert results[2]["status"] == "error"