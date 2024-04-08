from typing import Optional

import requests
from joblib import Memory
from pywikidata import Entity
from langdetect import detect

from app.config import DEFAULT_CACHE_PATH

memory = Memory(DEFAULT_CACHE_PATH, verbose=0)


@memory.cache
def label_to_entity_idx(label: str) -> Optional[str]:
    try:
        entities = [e.idx for e in Entity.from_label(label)]
    except:
        entities = None

    if entities is None:
        try:
            entities = [e.idx for e in Entity.from_label(label.capitalize())]
        except:
            entities = None
    
    if entities is None:
        try:
            entities = [e.idx for e in Entity.from_label(label.title())]
        except:
            entities = None
    
    if entities is None:
        return None

    idx = list(sorted(entities, key=lambda idx: int(idx[1:])))[0]
    return idx


def label_to_entity_idx_with_search(label: str) -> Optional[str]:
    full_match_result = label_to_entity_idx(label)
    if full_match_result is not None:
        return full_match_result

    search_results = get_wd_search_results(label, 1)
    if len(search_results) > 0:
        return search_results[0]
    else:
        return None


def is_valid_entity_idx(idx: str) -> bool:
    return len(idx) > 1 and idx[0].capitalize() == 'Q' and idx[1:].isdigit() and int(idx[1:]) > 0


def validate_or_search_entity_idx(idx: str) -> Optional[str]:
    return idx if is_valid_entity_idx(idx) else label_to_entity_idx_with_search(idx)


@memory.cache
def get_wd_search_results(
    search_string: str,
    max_results: int = 500,
    mediawiki_api_url: str = "https://www.wikidata.org/w/api.php",
    user_agent: str = None,
) -> list:
    params = {
        'action': 'wbsearchentities',
        'language': detect(search_string),
        'search': search_string,
        'format': 'json',
        'limit': 50
    }

    user_agent = "pywikidata" if user_agent is None else user_agent
    headers = {
        'User-Agent': user_agent
    }

    cont_count = 1
    results = []
    while cont_count > 0:
        params.update({'continue': 0 if cont_count == 1 else cont_count})

        reply = requests.get(mediawiki_api_url, params=params, headers=headers)
        reply.raise_for_status()
        search_results = reply.json()

        if search_results['success'] != 1:
            raise Exception('WD search failed')
        else:
            for i in search_results['search']:
                results.append(i['id'])

        if 'search-continue' not in search_results:
            cont_count = 0
        else:
            cont_count = search_results['search-continue']

        if cont_count > max_results:
            break

    return results
