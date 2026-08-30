from . import views
from unittest.mock import MagicMock


def test_index_smoke():
    request = MagicMock()
    response = views.index(request)
    assert response is not None
