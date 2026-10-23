# Copyright (c) 2024 Chen Liu
# All rights reserved.
import random
import time
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple

import requests

NO_AUTHOR_FOUND_STR = 'No_author_found'

# Observation: the Nominatim package is very bad at getting the geolocation of companies (geolocation of universities are fine).
# Temporary solution: hard code the geolocations of the companies.
# NOTE: The headquarter represents the whole company which usually has many offices accross the world.
KNOWN_AFFILIATION_DICT = {
    'amazon': ('King County', 'Seattle', 'Washington', 'USA', 47.622721, -122.337176),
    'meta': ('Menlo Park', 'San Mateo', 'California', 'USA', 37.4851, -122.1483),
    'facebook': ('Menlo Park', 'San Mateo', 'California', 'USA', 37.4851, -122.1483),
    'microsoft': ('King County', 'Redmond', 'Washington', 'USA', 47.645695, -122.131803),
    'ibm': ('Westchester', 'Armonk', 'New York', 'USA', 41.108252, -73.719887),
    'google': ('Santa Clara', 'Mountain View', 'California', 'USA', 37.421473, -122.080679),
    'deepmind': ('London', 'London', 'England', 'United Kingdom', 51.533802, -0.124872),
    'openai': ('San Francisco', 'San Francisco', 'California', 'USA', 37.762598, -122.414665),
    'nvidia': ('Santa Clara', 'Santa Clara', 'California', 'USA', 37.370955, -121.955237),
    'apple': ('Santa Clara', 'Cupertino', 'California', 'USA', 37.334789, -122.008965),
    'intel': ('Santa Clara', 'Santa Clara', 'California', 'USA', 37.387924, -121.963477),
    'tencent': ('Nanshan', 'Shenzhen', 'Guangdong', 'China', 22.540416, 113.934355),
    'alibaba': ('Yuhang', 'Hangzhou', 'Zhejiang', 'China', 30.279901, 120.023727),
    'bytedance': ('Haidian', 'Beijing', 'Beijing', 'China', 39.983624, 116.313408),
    'baidu': ('Haidian', 'Beijing', 'Beijing', 'China', 40.040239, 116.272669),
    'huawei': ('Longgang', 'Shenzhen', 'Guangdong', 'China', 22.657422, 114.057222),
    'samsung': ('Suwon', 'Suwon', 'Gyeonggi', 'South Korea', 37.258333, 127.054167),
    'morgan stanley': ('New York', 'New York', 'New York', 'USA', 40.760251, -73.98518),
    'siemens healthineers': ('Forchheim', 'Forchheim', 'Bavaria', 'Germany', 49.702088, 11.055870),
    'oracle': ('Travis', 'Austin', 'Texas', 'USA', 30.242913, -97.721641)
}

SCHOLAR_BASE = 'https://scholar.google.com'

# A small pool of modern desktop User-Agent strings. We rotate through these to
# look less like a single automated client, which reduces the chance of being
# throttled by Google Scholar.
_USER_AGENTS = [
    ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
     '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
    ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
     '(KHTML, like Gecko) Version/17.4 Safari/605.1.15'),
    ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
     '(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'),
    ('Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0'),
]


class CaptchaError(Exception):
    '''Raised when Google Scholar serves a CAPTCHA / "unusual traffic" page.

    Unlike the previous interactive Selenium flow, we do NOT block waiting for a
    human to solve a pop-up CAPTCHA (that is impossible in headless/server
    environments). Instead we raise this exception so the caller can checkpoint
    progress, back off, and resume later.
    '''
    pass


class _Fetcher:
    '''
    Thin wrapper around ``requests.Session`` that:
      * rotates User-Agent strings,
      * retries with exponential backoff on transient failures,
      * detects CAPTCHA / block pages and raises :class:`CaptchaError`,
      * enforces a minimum interval between requests to be polite.
    '''

    _CAPTCHA_MARKERS = (
        'unusual traffic', 'not a robot', '/sorry/', 'please show you',
        'id="gs_captcha', 'recaptcha', 'enablejs',
    )
    _BLOCK_MARKERS = ('access denied', 'forbidden')

    def __init__(self, min_interval: float = 1.0, timeout: int = 30):
        self.session = requests.Session()
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_request_time = 0.0

    def _headers(self) -> dict:
        return {
            'User-Agent': random.choice(_USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }

    def get(self, url: str, max_retries: int = 4) -> str:
        '''Fetch ``url`` and return the HTML, with retries and CAPTCHA detection.'''
        last_error = None
        for attempt in range(max_retries):
            # Politeness throttle: never hammer the server faster than min_interval.
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            # A little jitter on top, to avoid a fixed request cadence.
            time.sleep(random.uniform(0.3, 1.2))

            try:
                response = self.session.get(url, headers=self._headers(), timeout=self.timeout)
                self._last_request_time = time.time()
                html = response.text
                lower_html = html.lower()

                if any(marker in lower_html for marker in self._CAPTCHA_MARKERS):
                    # A CAPTCHA cannot be solved automatically here. Signal the caller.
                    raise CaptchaError(
                        'Google Scholar served a CAPTCHA / "unusual traffic" page for %s. '
                        'Progress has been checkpointed; wait a while (or use a proxy) and re-run '
                        'to resume from the cache.' % url)

                if response.status_code == 200 and not any(
                        marker in lower_html for marker in self._BLOCK_MARKERS):
                    return html

                last_error = 'HTTP %d' % response.status_code
            except CaptchaError:
                raise
            except Exception as e:  # network error, timeout, etc.
                last_error = repr(e)

            # Exponential backoff before the next attempt: 2s, 4s, 8s, ...
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))

        raise RuntimeError('Failed to fetch %s after %d attempts (last error: %s).' % (
            url, max_retries, last_error))


# A single module-level fetcher shared across calls so the session (and its
# politeness throttle) persists. ``min_interval`` can be tuned by the caller.
_fetcher = _Fetcher()


def configure_fetcher(min_interval: float = 1.0, proxies: Optional[dict] = None) -> None:
    '''Reconfigure the shared fetcher (request interval and optional proxies).'''
    global _fetcher
    _fetcher = _Fetcher(min_interval=min_interval)
    if proxies:
        _fetcher.session.proxies.update(proxies)


def get_publications_and_cites(scholar_id: str) -> Tuple[str, List[Tuple[str, str]]]:
    '''
    Fetch the author's profile and return ``(author_name, [(cites_id, paper_title), ...])``.

    This parses the profile page(s) directly, extracting each publication's
    "Cited by" link (which carries the ``cites`` id). Compared with the old
    approach of calling ``scholarly.fill`` on every publication, this needs only
    a handful of requests (one per 100 publications), greatly reducing the chance
    of being blocked.
    '''
    author_name = None
    cites_list = []
    page_size = 100
    cstart = 0
    while True:
        url = '%s/citations?user=%s&hl=en&cstart=%d&pagesize=%d' % (
            SCHOLAR_BASE, scholar_id, cstart, page_size)
        html = _fetcher.get(url)
        soup = BeautifulSoup(html, 'html.parser')

        if author_name is None:
            name_tag = soup.find('div', id='gsc_prf_in')
            author_name = name_tag.text.strip() if name_tag else scholar_id

        rows = soup.find_all('tr', class_='gsc_a_tr')
        for row in rows:
            title_tag = row.find('a', class_='gsc_a_at')
            paper_title = title_tag.text.strip() if title_tag else ''
            cited_by_link = row.find('a', class_='gsc_a_ac')
            if cited_by_link is not None:
                href = cited_by_link.get('href', '')
                if 'cites=' in href:
                    # A publication can have multiple comma-separated cites ids.
                    cites_value = href.split('cites=')[1].split('&')[0]
                    for cites_id in cites_value.split(','):
                        cites_id = cites_id.strip()
                        if cites_id:
                            cites_list.append((cites_id, paper_title))

        # Stop when the last page had fewer than a full page of rows.
        if len(rows) < page_size:
            break
        cstart += page_size

    return author_name, cites_list


def get_citing_author_ids_and_citing_papers(paper_url: str) -> List[Tuple[str, str]]:
    '''
    Find the (Google Scholar IDs of authors, titles of papers) who cite a given paper on Google Scholar.

    Parameters
    --------
    paper_url: URL of the paper BEING cited (``https://scholar.google.com/scholar?...&cites=<id>``).
    '''
    citing_authors_and_citing_papers = []

    # Google Scholar paginates citing results 10 per page via the ``start`` param.
    start = 0
    page_size = 10
    max_pages = 100  # Safety cap (Scholar itself caps at ~1000 results).
    for _ in range(max_pages):
        sep = '&' if '?' in paper_url else '?'
        url = '%s%sstart=%d' % (paper_url, sep, start)
        html = _fetcher.get(url)
        soup = BeautifulSoup(html, 'html.parser')

        page_results = _parse_citation_results_page(soup)
        if not page_results:
            break
        citing_authors_and_citing_papers += page_results

        # Is there a next page? Scholar renders numbered navigation buttons.
        if not _has_next_page(soup, start, page_size):
            break
        start += page_size

    return citing_authors_and_citing_papers


def _parse_citation_results_page(soup) -> List[Tuple[str, str]]:
    '''Parse one page of citing results into (author_id, paper_title) tuples.'''
    results = []
    for result in soup.find_all('div', class_='gs_ri'):
        title_tag = result.find('h3', class_='gs_rt')
        if not title_tag:
            continue
        title = title_tag.get_text().replace('[HTML]', '').replace('[PDF]', '').strip()
        author_found = False
        for link in result.find_all('a', href=True):
            if 'user=' in link['href']:
                author_id = link['href'].split('user=')[1].split('&')[0]
                results.append((author_id, title))
                author_found = True
        if not author_found:
            results.append((NO_AUTHOR_FOUND_STR, title))
    return results


def _has_next_page(soup, current_start: int, page_size: int) -> bool:
    '''Return True if a "next page" navigation link exists past current_start.'''
    # The cleanest signal is the dedicated next-page button.
    next_button = soup.find('td', {'align': 'left'})
    if next_button and next_button.find('a'):
        return True
    # Fallback: numbered navigation buttons.
    current_page = current_start // page_size + 1
    for navigation in soup.find_all('a', class_='gs_nma'):
        text = navigation.text
        if text and text.isnumeric() and int(text) == current_page + 1:
            return True
    return False


def get_author_name_and_affiliation(author_id: str,
                                    conservative: bool = False) -> Optional[Tuple[str, str]]:
    '''
    Fetch an author's profile and return ``(name, affiliation_string)``.

    If ``conservative`` is True, only the *verified* organization (the linked
    institution in the affiliation line) is returned; authors without a verified
    organization link yield ``None``. Otherwise the full self-reported
    affiliation string is returned (higher recall).
    '''
    if author_id == NO_AUTHOR_FOUND_STR:
        return None

    url = '%s/citations?user=%s&hl=en' % (SCHOLAR_BASE, author_id)
    html = _fetcher.get(url)
    soup = BeautifulSoup(html, 'html.parser')

    name_tag = soup.find('div', id='gsc_prf_in')
    name = name_tag.text.strip() if name_tag else None
    if not name:
        return None

    affiliation_tag = soup.find('div', class_='gsc_prf_il')
    if affiliation_tag is None:
        return None

    if conservative:
        # Only trust the linked (verified) organization name(s).
        links = [a.text.strip() for a in affiliation_tag.find_all('a')
                 if a.get('href', '').find('view_org') != -1 or 'org=' in a.get('href', '')]
        if not links:
            return None
        return (name, ' | '.join(links))

    return (name, affiliation_tag.text.strip())
