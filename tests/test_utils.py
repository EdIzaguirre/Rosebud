from unittest.mock import Mock, patch
from ..utils import get_id_list, get_data, write_file
import os
from dotenv import load_dotenv
load_dotenv()

TMBD_API_KEY = os.getenv('TMBD_API_KEY')


@patch('requests.get')
def test_get_id_list(mock_get):
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {'results': [
        {'id': '1'}, {'id': '2'}, {'id': '3'}, {'id': '4'}, {'id': '5'}]}
    year = 2020
    ids = get_id_list(TMBD_API_KEY, year)
    assert isinstance(ids, list)
    assert all(isinstance(i, str) for i in ids)


@patch('requests.get')
def test_get_data(mock_get, my_movie):
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = my_movie
    movie_id = '1234'
    data = get_data(TMBD_API_KEY, movie_id)
    assert isinstance(data, dict)
    assert 'title' in data
    assert 'runtime' in data
    assert 'original_language' in data
    assert 'release_date' in data
    assert 'overview' in data
    assert 'genres' in data
    assert 'production_companies' in data


def test_write_file(tmp_path, my_movie):
    filename = tmp_path / "test.csv"
    write_file(filename, my_movie)
    with open(filename, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert lines[0].strip().split(',') == [
        'Test Movie', '120', 'English', 'This is a test movie.', '2020',
        'Drama', 'Test Keyword', 'Test Actor Name', 'Test Director Name',
        'Test Stream Service ', 'Test Buy Service ', 'Test Rent Service ',
        'Test Studio']
