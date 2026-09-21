from experiment3_agent.video_tools import uniform_indices


def test_uniform_indices():
    assert uniform_indices(0, 10, 3) == [0, 5, 10]
    assert uniform_indices(5, 5, 1) == [5]
