import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.core.security import get_current_user
from src.api.main import app


def test_feedback_requires_auth(client: TestClient) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post(
        "/api/v1/feedback",
        json={"prediction_id": str(uuid.uuid4()), "signal": "correct"},
    )
    assert response.status_code == 401


def test_feedback_returns_404_for_unknown_prediction(client: TestClient) -> None:
    with patch(
        "src.api.repositories.feedback.FeedbackRepository.prediction_exists",
        new_callable=AsyncMock,
        return_value=False,
    ):

        response = client.post(
            "/api/v1/feedback",
            json={"prediction_id": str(uuid.uuid4()), "signal": "correct"},
        )
    assert response.status_code == 404
    assert response.json()["error_code"] == "NOT_FOUND"


def test_feedback_returns_200_on_success(client: TestClient) -> None:
    with (
        patch(
            "src.api.repositories.feedback.FeedbackRepository.prediction_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.api.repositories.feedback.FeedbackRepository.create",
            new_callable=AsyncMock,
        ),
    ):

        response = client.post(
            "/api/v1/feedback",
            json={
                "prediction_id": str(uuid.uuid4()),
                "signal": "correct",
                "comment": "Looks good.",
            },
        )
    assert response.status_code == 200
    assert "message" in response.json()


def test_feedback_stores_null_comment_when_absent(client: TestClient) -> None:
    create_mock = AsyncMock()
    with (
        patch(
            "src.api.repositories.feedback.FeedbackRepository.prediction_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.api.repositories.feedback.FeedbackRepository.create",
            create_mock,
        ),
    ):

        client.post(
            "/api/v1/feedback",
            json={"prediction_id": str(uuid.uuid4()), "signal": "incorrect"},
        )
    # comment kwarg must be None when not provided
    _, kwargs = create_mock.call_args
    assert kwargs.get("comment") is None


def test_feedback_allows_multiple_submissions(client: TestClient) -> None:
    """Two submissions for the same prediction must both succeed."""
    prediction_id = str(uuid.uuid4())
    with (
        patch(
            "src.api.repositories.feedback.FeedbackRepository.prediction_exists",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.api.repositories.feedback.FeedbackRepository.create",
            new_callable=AsyncMock,
        ),
    ):

        r1 = client.post(
            "/api/v1/feedback",
            json={"prediction_id": prediction_id, "signal": "correct"},
        )
        r2 = client.post(
            "/api/v1/feedback",
            json={"prediction_id": prediction_id, "signal": "incorrect"},
        )
    assert r1.status_code == 200
    assert r2.status_code == 200
