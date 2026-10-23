# Copyright (c) 2024 Chen Liu
# All rights reserved.
'''
OpenAlex-based citation data source.

Google Scholar's citing-paper search endpoint (``/scholar?cites=...``) is
heavily CAPTCHA-protected and often unusable from servers / shared IPs. OpenAlex
(https://openalex.org) exposes the same information through a proper, free API
with no CAPTCHA, and its institution records even carry latitude/longitude, so no
separate geocoding step is needed.

Pipeline:
  1. Take the author's publication titles (e.g. from their Google Scholar
     profile, whose profile endpoint is not blocked) and match each to an
     OpenAlex work, verifying the author's name appears on the matched work.
  2. For each work, fetch its citing works.
  3. For each citing work, read the authors and their institutions.
  4. Resolve each institution to coordinates via OpenAlex's ``geo`` field.
'''
import time
import requests
from typing import Dict, List, Optional, Tuple

OPENALEX_BASE = 'https://api.openalex.org'


class OpenAlexClient:
    '''Minimal OpenAlex API client with API key, polite pool, and retries.'''

    def __init__(self, api_key: Optional[str] = None, mailto: Optional[str] = None,
                 min_interval: float = 0.1, timeout: int = 30):
        self.api_key = api_key
        self.mailto = mailto
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = requests.Session()
        self._last_request_time = 0.0

    def get(self, path: str, params: Optional[dict] = None, max_retries: int = 5) -> dict:
        params = dict(params or {})
        if self.api_key:
            params['api_key'] = self.api_key
        if self.mailto:
            params['mailto'] = self.mailto
        url = OPENALEX_BASE + path

        last_error = None
        for attempt in range(max_retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                self._last_request_time = time.time()
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 429:
                    # Rate limited: back off more aggressively.
                    last_error = 'HTTP 429 (rate limited)'
                    time.sleep(5 * (attempt + 1))
                    continue
                last_error = 'HTTP %d: %s' % (response.status_code, response.text[:200])
            except Exception as e:
                last_error = repr(e)
            time.sleep(2 ** attempt)
        raise RuntimeError('OpenAlex request failed for %s (last error: %s)' % (url, last_error))


def _last_name(full_name: str) -> str:
    parts = [p for p in full_name.replace('.', ' ').split() if p]
    return parts[-1].lower() if parts else ''


def match_works_by_titles(client: OpenAlexClient,
                          titles: List[str],
                          author_name_hint: Optional[str] = None) -> List[Tuple[str, str]]:
    '''
    Match each publication title to an OpenAlex work.

    Returns a list of ``(openalex_work_id, matched_title)``. When
    ``author_name_hint`` is given, a candidate is preferred only if the author's
    last name appears among the work's authors, which guards against matching a
    different paper with a similar title.
    '''
    hint_last = _last_name(author_name_hint) if author_name_hint else None
    work_map = {}  # work_id -> matched title (the user's own paper title)
    for title in titles:
        clean_title = title.replace('[HTML]', '').replace('[PDF]', '').strip()
        if not clean_title:
            continue
        try:
            data = client.get('/works', params={
                'filter': 'title.search:%s' % clean_title,
                'select': 'id,display_name,authorships',
                'per-page': 5,
            })
        except Exception:
            continue
        results = data.get('results', [])
        if not results:
            continue

        chosen = None
        if hint_last:
            for w in results:
                authors = [a.get('author', {}).get('display_name', '') for a in w.get('authorships', [])]
                if any(hint_last == _last_name(a) for a in authors):
                    chosen = w
                    break
        if chosen is None:
            # Fall back to the top hit only if no author hint was provided.
            if hint_last is None:
                chosen = results[0]
        if chosen is not None:
            work_id = chosen['id'].split('/')[-1]
            work_map.setdefault(work_id, clean_title)
    return list(work_map.items())


def get_citing_records(client: OpenAlexClient,
                       work_id: str,
                       cited_title: str) -> List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]:
    '''
    Fetch all works citing ``work_id``.

    Returns a list of
    ``(citing_author_name, citing_paper_title, cited_title, institution_id,
       institution_name, country_code)``.
    One record per (citing author, institution) pair.
    '''
    records = []
    cursor = '*'
    while cursor:
        data = client.get('/works', params={
            'filter': 'cites:%s' % work_id,
            'select': 'id,display_name,authorships',
            'per-page': 200,
            'cursor': cursor,
        })
        for cw in data.get('results', []):
            citing_title = (cw.get('display_name') or '').strip()
            for authorship in cw.get('authorships', []):
                author_name = authorship.get('author', {}).get('display_name', '')
                institutions = authorship.get('institutions', [])
                if not institutions:
                    records.append((author_name, citing_title, cited_title, None, None, None))
                else:
                    for ins in institutions:
                        inst_id = ins.get('id', '').split('/')[-1] if ins.get('id') else None
                        records.append((author_name, citing_title, cited_title,
                                        inst_id, ins.get('display_name'), ins.get('country_code')))
        cursor = data.get('meta', {}).get('next_cursor')
    return records


def resolve_institution_geo(client: OpenAlexClient,
                            institution_ids: List[str]) -> Dict[str, dict]:
    '''
    Resolve OpenAlex institution ids to geo info.

    Returns ``{institution_id: {display_name, country, latitude, longitude, city, region}}``.
    Institutions are fetched in batches via the ``ids.openalex`` OR filter.
    '''
    geo_map = {}
    unique_ids = [iid for iid in dict.fromkeys(institution_ids) if iid]
    batch_size = 50
    for start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[start:start + batch_size]
        try:
            data = client.get('/institutions', params={
                'filter': 'ids.openalex:%s' % '|'.join(batch),
                'select': 'id,display_name,country_code,geo',
                'per-page': batch_size,
            })
        except Exception:
            continue
        for ins in data.get('results', []):
            iid = ins['id'].split('/')[-1]
            geo = ins.get('geo') or {}
            geo_map[iid] = {
                'display_name': ins.get('display_name'),
                'country': geo.get('country'),
                'country_code': ins.get('country_code'),
                'latitude': geo.get('latitude'),
                'longitude': geo.get('longitude'),
                'city': geo.get('city'),
                'region': geo.get('region'),
            }
    return geo_map
