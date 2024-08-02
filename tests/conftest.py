import pytest


@pytest.fixture
def my_movie():
    data = {
        'title': 'Test Movie',
        'runtime': 120,
        'original_language': 'en',
        'id': '1234',
        'release_date': '2020-01-01',
        'overview': 'This is a test movie.',
        'genres': [{'name': 'Drama'}],
        'production_companies': [{'name': 'Test Studio'}],
        'keywords': {'keywords': [{'name': 'Test Keyword'}]},
        'watch/providers': {'results':
                            {'US':
                             {'rent': [{'provider_name': 'Test Rent Service'}],
                              'buy': [{'provider_name': 'Test Buy Service'}],
                              'flatrate': [
                                  {'provider_name': 'Test Stream Service'}]
                              }
                             }
                            },
        'credits': {'cast': [{'name': 'Test Actor Name'}],
                    'crew': [{'job': 'Director',
                              'name': 'Test Director Name'}]},


    }

    return data
