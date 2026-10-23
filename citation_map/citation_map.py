# Copyright (c) 2024 Chen Liu
# All rights reserved.
import folium
import pandas as pd
import os
import pickle
import pycountry
import re
import random
import time

from geopy.geocoders import Nominatim, ArcGIS
from tqdm import tqdm
from typing import Any, List, Tuple, Optional

from .scholarly_support import (
    CaptchaError,
    configure_fetcher,
    get_publications_and_cites,
    get_citing_author_ids_and_citing_papers,
    get_author_name_and_affiliation,
    NO_AUTHOR_FOUND_STR,
    KNOWN_AFFILIATION_DICT,
)
from .openalex_support import (
    OpenAlexClient,
    match_works_by_titles,
    get_citing_records,
    resolve_institution_geo,
)


def find_all_citing_authors(scholar_id: str,
                            cache_folder: Optional[str] = None) -> List[Tuple[str]]:
    '''
    Step 1. Find all publications of the given Google Scholar ID (and their `cites` ids).
    Step 2. Find all citing authors.

    Both steps are checkpointed: Step 1 is cached as a whole, and Step 2 is cached
    incrementally per `cites_id`, so a CAPTCHA or crash can be resumed by simply
    re-running (already-fetched citation pages are loaded from cache).
    '''
    # ---- Step 1: publications + cites ids (one cheap checkpoint). ----
    pubs_cache = _cache_path(cache_folder, scholar_id, 'publications_and_cites.pkl')
    if pubs_cache is not None and os.path.exists(pubs_cache):
        author_name, cites_list = load_cache(pubs_cache)
        print('Loaded profile from cache: %d publications with citations.\n' % len(cites_list))
    else:
        author_name, cites_list = get_publications_and_cites(scholar_id)
        print('Author profile found: %s. %d publications have citations.\n' % (
            author_name, len(cites_list)))
        if pubs_cache is not None:
            save_cache((author_name, cites_list), pubs_cache)

    # ---- Step 2: citing authors per cites_id (incremental checkpoint). ----
    citing_cache = _cache_path(cache_folder, scholar_id, 'citing_authors_by_cites.pkl')
    citing_by_cites = {}
    if citing_cache is not None and os.path.exists(citing_cache):
        citing_by_cites = load_cache(citing_cache)
        print('Resuming: %d/%d citation pages already fetched.\n' % (
            len(citing_by_cites), len(cites_list)))

    try:
        for cites_id, cited_paper_title in tqdm(
                cites_list,
                desc='Finding citing authors on your %d cited publications' % len(cites_list),
                total=len(cites_list)):
            if cites_id in citing_by_cites:
                continue
            citing_paper_search_url = 'https://scholar.google.com/scholar?hl=en&cites=' + cites_id
            citing = get_citing_author_ids_and_citing_papers(citing_paper_search_url)
            citing_by_cites[cites_id] = [
                (citing_author_id, citing_paper_title, cited_paper_title)
                for citing_author_id, citing_paper_title in citing
            ]
            if citing_cache is not None:
                save_cache(citing_by_cites, citing_cache)
    except CaptchaError as e:
        # Checkpoint is already on disk; surface a clear, actionable message.
        if citing_cache is not None:
            save_cache(citing_by_cites, citing_cache)
        print('\n[CAPTCHA] %s' % e)
        print('Partial progress saved (%d/%d citation pages). Re-run later to resume.\n' % (
            len(citing_by_cites), len(cites_list)))
        raise

    all_citing_author_paper_tuple_list = []
    for entries in citing_by_cites.values():
        all_citing_author_paper_tuple_list.extend(entries)
    return all_citing_author_paper_tuple_list


def find_all_citing_affiliations(all_citing_author_paper_tuple_list: List[Tuple[str]],
                                 affiliation_conservative: bool = False,
                                 scholar_id: Optional[str] = None,
                                 cache_folder: Optional[str] = None) -> List[Tuple[str]]:
    '''
    Step 3. Find all citing affiliations.

    Each unique citing author is fetched at most once, and the result is cached
    incrementally per author id so the step is resumable.
    '''
    # Cache of author_id -> (name, affiliation_string). None means "no affiliation".
    aff_cache_path = _cache_path(cache_folder, scholar_id, 'affiliation_by_author.pkl')
    affiliation_by_author = {}
    if aff_cache_path is not None and os.path.exists(aff_cache_path):
        affiliation_by_author = load_cache(aff_cache_path)

    # Unique author ids (keep one representative paper pairing per occurrence).
    unique_author_ids = []
    seen = set()
    for citing_author_id, _, _ in all_citing_author_paper_tuple_list:
        if citing_author_id not in seen:
            seen.add(citing_author_id)
            unique_author_ids.append(citing_author_id)

    try:
        for citing_author_id in tqdm(
                unique_author_ids,
                desc='Finding affiliations of %d unique citing authors' % len(unique_author_ids),
                total=len(unique_author_ids)):
            if citing_author_id in affiliation_by_author:
                continue
            if citing_author_id == NO_AUTHOR_FOUND_STR:
                affiliation_by_author[citing_author_id] = None
                continue
            result = get_author_name_and_affiliation(
                citing_author_id, conservative=affiliation_conservative)
            affiliation_by_author[citing_author_id] = result
            if aff_cache_path is not None:
                save_cache(affiliation_by_author, aff_cache_path)
    except CaptchaError as e:
        if aff_cache_path is not None:
            save_cache(affiliation_by_author, aff_cache_path)
        print('\n[CAPTCHA] %s' % e)
        print('Partial progress saved (%d/%d authors). Re-run later to resume.\n' % (
            len([v for v in affiliation_by_author.values()]), len(unique_author_ids)))
        raise

    # Build the (author_name, citing_paper, cited_paper, affiliation) tuples.
    author_paper_affiliation_tuple_list = []
    for citing_author_id, citing_paper_title, cited_paper_title in all_citing_author_paper_tuple_list:
        if citing_author_id == NO_AUTHOR_FOUND_STR:
            author_paper_affiliation_tuple_list.append(
                (NO_AUTHOR_FOUND_STR, citing_paper_title, cited_paper_title, NO_AUTHOR_FOUND_STR))
            continue
        result = affiliation_by_author.get(citing_author_id)
        if result is None:
            continue
        author_name, affiliation_string = result
        author_paper_affiliation_tuple_list.append(
            (author_name, citing_paper_title, cited_paper_title, affiliation_string))

    return author_paper_affiliation_tuple_list


def clean_affiliation_names(author_paper_affiliation_tuple_list: List[Tuple[str]]) -> List[Tuple[str]]:
    '''
    Optional Step. Clean up the names of affiliations from the authors' affiliation tab on their Google Scholar profiles.
    NOTE: This logic is very naive. Please send an issue or pull request if you have any idea how to improve it.
    Currently we will not consider any paid service or tools that pose extra burden on the users, such as GPT API.
    '''
    cleaned_author_paper_affiliation_tuple_list = []
    for author_name, citing_paper_title, cited_paper_title, affiliation_string in author_paper_affiliation_tuple_list:
        if author_name == NO_AUTHOR_FOUND_STR:
            cleaned_author_paper_affiliation_tuple_list.append((NO_AUTHOR_FOUND_STR, citing_paper_title, cited_paper_title, NO_AUTHOR_FOUND_STR))
        else:
            # Use a regular expression to split the string by '|', ';', '/' or 'and'.
            # NOTE: '|' is common on Scholar when an author lists multiple affiliations
            # (e.g. "University of Cambridge | Sun Yat-sen University"); without splitting
            # on it the whole string fails to geocode.
            substring_list = [part.strip() for part in re.split(r'[;|/]|\band\b', affiliation_string)]
            # Further split the substrings by ',' if the latter component is not a country.
            substring_list = __country_aware_comma_split(substring_list)

            for substring in substring_list:
                # Use a regular expression to remove anything before 'at', or '@'.
                cleaned_affiliation = re.sub(r'.*?\bat\b|.*?@', '', substring, flags=re.IGNORECASE).strip()
                # Use a regular expression to filter out strings that represent
                # a person's identity rather than affiliation.
                is_common_identity_string = re.search(
                    re.compile(
                        r'\b(director|manager|chair|engineer|programmer|scientist|professor|lecturer|phd|ph\.d|postdoc|doctor|student|department of)\b',
                        re.IGNORECASE),
                    cleaned_affiliation)
                if not is_common_identity_string and cleaned_affiliation:
                    cleaned_author_paper_affiliation_tuple_list.append((author_name, citing_paper_title, cited_paper_title, cleaned_affiliation))
    return cleaned_author_paper_affiliation_tuple_list

def fill_known_affiliations(affiliation_name: str) -> Optional[str]:
    '''
    If the affiliation is known, return its geolocation.
    If not, return None.
    The reason we have this function is taht geolocator may return hilarious results,
    such as putting the company Amazon in Amazon rain forest.
    NOTE: This is a temporal fix. Can be replaced by smarter natural language parsers.
    '''
    for key in KNOWN_AFFILIATION_DICT:
        if key in affiliation_name.lower():
            return KNOWN_AFFILIATION_DICT[key]
    return None

# Country name aliases, used to merge different spellings of the same country
# (e.g. the hard-coded 'USA' entries vs. Nominatim's 'United States') so the
# country count is not inflated.
_COUNTRY_ALIASES = {
    'usa': 'United States',
    'us': 'United States',
    'u.s.': 'United States',
    'u.s.a.': 'United States',
    'united states of america': 'United States',
    'america': 'United States',
    'uk': 'United Kingdom',
    'u.k.': 'United Kingdom',
    'england': 'United Kingdom',
    'scotland': 'United Kingdom',
    'wales': 'United Kingdom',
    'prc': 'China',
    "people's republic of china": 'China',
}

# Honorific / membership / generic tokens that are NOT institutions. An
# affiliation made up entirely of these is junk (e.g. 'IEEE SM' = "IEEE Senior
# Member", 'FIEEE' = "IEEE Fellow"), and geocoding it produces nonsense such as
# matching the token 'SM' to the country code of San Marino.
_HONORIFIC_TOKENS = {
    'ieee', 'aaai', 'acm', 'siam', 'asme', 'aps', 'rsc', 'acs',
    'fellow', 'member', 'senior', 'life', 'distinguished', 'emeritus',
    'sm', 'sr', 'jr', 'fieee', 'smieee', 'mieee',
    'phd', 'msc', 'bsc', 'md', 'student', 'candidate', 'postdoc', 'postdoctoral',
    'professor', 'prof', 'assistant', 'associate', 'adjunct', 'lecturer',
    'researcher', 'scientist', 'scholar', 'director', 'head', 'chair',
    'msca', 'mitacs', 'phd.', 'ph.d', 'ph.d.',
    'and', 'of', 'the', 'at', 'in', 'for', 'a', '&',
}


def normalize_country(country_name: Optional[str]) -> Optional[str]:
    '''Map different spellings of the same country to one canonical name.'''
    if country_name is None or (isinstance(country_name, float)):
        return country_name
    name = str(country_name).strip()
    if not name:
        return name
    return _COUNTRY_ALIASES.get(name.lower(), name)


def affiliation_invalid(affiliation_name: str) -> bool:
    '''
    Check if the affiliation is invalid.
    Typical invalid affiliation contains non-affiliation words, such as 'computer science'.
    Invalid affiliations will waste time in geolocator.geocode(affiliation_name).
    NOTE: This is a temporal fix. Can be replaced by smarter natural language parsers.
    '''
    invalid_affiliation_set = {
        NO_AUTHOR_FOUND_STR.lower(),
        'computer', 'computer science', 'electrical', 'engineering', 'researcher',
        'scholar', 'inc.', 'school', 'department', 'student', 'candidate', 'professor', 'faculty', 'associate',
        'telecommunications',
    }
    for key in invalid_affiliation_set:
        if key in affiliation_name.lower():
            return True

    # Reject strings made up entirely of honorific / membership / generic tokens
    # (e.g. 'IEEE SM', 'FIEEE', 'Senior Member'), which otherwise geocode to junk.
    tokens = [t for t in re.split(r'[^a-zA-Z.&]+', affiliation_name.lower()) if t]
    if tokens and all(token in _HONORIFIC_TOKENS for token in tokens):
        return True

    # Reject raw strings that still carry a society membership marker mixed in
    # (e.g. "University of Arizona, AAAI SM, IEEE SM"): the geocoder latches onto
    # the "SM" token and returns San Marino. The cleaned sub-affiliations already
    # cover the real institution in these strings, so dropping the raw form is safe.
    if re.search(r'\b(aaai|ieee|acm|siam|asme|aps)\s+(sm|sr|fellow|member)\b', affiliation_name.lower()):
        return True
    if re.search(r'\b(fieee|smieee|mieee|faaai)\b', affiliation_name.lower()):
        return True

    return False


def _reverse_geocode_metadata(geolocator, latitude, longitude):
    '''Reverse-geocode coordinates into (county, city, state, country), best-effort.'''
    county, city, state, country = None, None, None, None
    try:
        location_metadata = geolocator.reverse('%s,%s' % (latitude, longitude), language='en')
        address = location_metadata.raw.get('address', {})
        county = address.get('county')
        city = address.get('city') or address.get('town') or address.get('village')
        state = address.get('state')
        country = normalize_country(address.get('country'))
    except Exception:
        pass
    return county, city, state, country


def affiliation_text_to_geocode(author_paper_affiliation_tuple_list: List[Tuple[str]],
                                max_attempts: int = 3,
                                scholar_id: Optional[str] = None,
                                cache_folder: Optional[str] = None) -> List[Tuple[str]]:
    '''
    Step 4: Convert affiliations in plain text to Geocode.

    Improvements over the original:
      * A disk cache of geocode results, so re-runs are instant and we never
        re-query the same affiliation twice.
      * Respect the Nominatim usage policy of <= 1 request/second.
      * Fall back to the ArcGIS geocoder when Nominatim cannot locate a string.
    '''
    coordinates_and_info = []
    # NOTE: According to the Nominatim Usage Policy (https://operations.osmfoundation.org/policies/nominatim/),
    # we are explicitly asked not to submit bulk requests on multiple threads.
    # Therefore, we will keep it to a loop instead of multiprocessing, and rate-limit to 1 req/s.
    geolocator = Nominatim(user_agent='citation_mapper', timeout=20)
    fallback_geolocator = ArcGIS(timeout=20)

    # Disk cache: affiliation_name -> (county, city, state, country, lat, lon) or None.
    geo_cache_path = _cache_path(cache_folder, scholar_id, 'geocode_cache.pkl')
    geo_cache = {}
    if geo_cache_path is not None and os.path.exists(geo_cache_path):
        geo_cache = load_cache(geo_cache_path)

    # Find unique affiliations and record their corresponding entries.
    affiliation_map = {}
    for entry_idx, (_, _, _, affiliation_name) in enumerate(author_paper_affiliation_tuple_list):
        affiliation_map.setdefault(affiliation_name, []).append(entry_idx)

    num_total_affiliations = len(affiliation_map)
    num_located_affiliations = 0
    last_query_time = 0.0

    def _record(affiliation_name, geo_location_tuple):
        '''geo_location_tuple = (county, city, state, country, lat, lon) or None.'''
        for entry_idx in affiliation_map[affiliation_name]:
            author_name, citing_paper_title, cited_paper_title, aff = author_paper_affiliation_tuple_list[entry_idx]
            if geo_location_tuple is None:
                coordinates_and_info.append((author_name, citing_paper_title, cited_paper_title, aff,
                                             '', '', '', '', '', ''))
            else:
                county, city, state, country, latitude, longitude = geo_location_tuple
                coordinates_and_info.append((author_name, citing_paper_title, cited_paper_title, aff,
                                             latitude, longitude, county, city, state, normalize_country(country)))

    for affiliation_name in tqdm(affiliation_map,
                                 desc='Finding geographic coordinates from %d unique citing affiliations in %d entries' % (
                                     len(affiliation_map), len(author_paper_affiliation_tuple_list)),
                                 total=len(affiliation_map)):
        # 0) Invalid (non-affiliation) string: record but do not geocode.
        #    Checked before the cache so that improved filters also drop stale
        #    junk that an earlier run may have cached (e.g. 'IEEE SM').
        if affiliation_invalid(affiliation_name):
            geo_cache[affiliation_name] = None
            _record(affiliation_name, None)
            continue

        # 1) Known affiliation (hard-coded headquarters). Checked before the cache
        #    so improved/corrected entries override stale cached geocodes
        #    (e.g. 'Allen Institute for AI' wrongly cached as Anguilla).
        known = fill_known_affiliations(affiliation_name)
        if known is not None:
            county, city, state, country, latitude, longitude = known
            result = (county, city, state, normalize_country(country), latitude, longitude)
            geo_cache[affiliation_name] = result
            _record(affiliation_name, result)
            num_located_affiliations += 1
            continue

        # 2) Cached result?
        if affiliation_name in geo_cache:
            result = geo_cache[affiliation_name]
            _record(affiliation_name, result)
            if result is not None:
                num_located_affiliations += 1
            continue

        # 3) Geocode via Nominatim, then ArcGIS as a fallback.
        result = None
        for _ in range(max_attempts):
            try:
                # Rate limit to <= 1 request/second for Nominatim.
                elapsed = time.time() - last_query_time
                if elapsed < 1.1:
                    time.sleep(1.1 - elapsed)
                geo_location = geolocator.geocode(affiliation_name)
                last_query_time = time.time()
                if geo_location is None:
                    geo_location = fallback_geolocator.geocode(affiliation_name)
                    last_query_time = time.time()
                if geo_location is not None:
                    county, city, state, country = _reverse_geocode_metadata(
                        geolocator, geo_location.latitude, geo_location.longitude)
                    last_query_time = time.time()
                    result = (county, city, state, country,
                              geo_location.latitude, geo_location.longitude)
                    break
            except Exception:
                time.sleep(1.0)
                continue

        geo_cache[affiliation_name] = result
        if geo_cache_path is not None:
            save_cache(geo_cache, geo_cache_path)
        _record(affiliation_name, result)
        if result is not None:
            num_located_affiliations += 1

    print('\nConverted %d/%d affiliations to Geocodes.' % (num_located_affiliations, num_total_affiliations))
    coordinates_and_info = [item for item in coordinates_and_info if item is not None]  # Filter out empty entries.
    return coordinates_and_info

def export_dict_to_csv(coordinates_and_info: List[Tuple[str]], csv_output_path: str) -> None:
    '''
    Step 5.1: Export csv file recording citation information.
    '''

    citation_df = pd.DataFrame(coordinates_and_info,
                               columns=['citing author name', 'citing paper title', 'cited paper title',
                                        'affiliation', 'latitude', 'longitude',
                                        'county', 'city', 'state', 'country'])

    citation_df.to_csv(csv_output_path)
    return

def read_csv_to_dict(csv_path: str) -> None:
    '''
    Step 5.1: Read csv file recording citation information.
    Only relevant if `read_from_csv` is True.
    '''

    citation_df = pd.read_csv(csv_path, index_col=0)
    coordinates_and_info = list(citation_df.itertuples(index=False, name=None))
    return coordinates_and_info

def create_map(coordinates_and_info: List[Tuple[str]], pin_colorful: bool = True):
    '''
    Step 5.2: Create the Citation World Map.

    For authors under the same affiliations, they will be displayed in the same pin.
    '''
    # NOTE: Use Esri's free (no API key) light-gray basemap. folium's default
    # OpenStreetMap tiles now return HTTP 403 ("tile usage policy"), and the
    # built-in CartoDB tiles now require an API key ("API KEY REQUIRED"); both
    # make the base map render as a wall of error tiles. Esri's ArcGIS Online
    # tiles render reliably without a key. (Esri uses {z}/{y}/{x} tile order.)
    citation_map = folium.Map(
        location=[20, 0], zoom_start=2,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ')

    # Find unique affiliations and record their corresponding entries.
    affiliation_map = {}
    for entry_idx, (_, _, _, affiliation_name, _, _, _, _, _, _) in enumerate(coordinates_and_info):
        if affiliation_name == NO_AUTHOR_FOUND_STR:
            continue
        elif affiliation_name not in affiliation_map.keys():
            affiliation_map[affiliation_name] = [entry_idx]
        else:
            affiliation_map[affiliation_name].append(entry_idx)

    if pin_colorful:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                  'darkpurple', 'pink', 'lightblue', 'lightgreen',
                  'gray', 'black', 'lightgray']
        for affiliation_name in affiliation_map:
            color = random.choice(colors)
            corresponding_entries = affiliation_map[affiliation_name]
            author_name_list = []
            location_valid = True
            for entry_idx in corresponding_entries:
                author_name, _, _, _, lat, lon, _, _, _, _  = coordinates_and_info[entry_idx]
                if pd.isna(lat) or pd.isna(lon) or lat == '' or lon == '':
                    location_valid = False
                author_name_list.append(author_name)
            if location_valid:
                folium.Marker([lat, lon], popup='%s (%s)' % (affiliation_name, ' & '.join(author_name_list)),
                            icon=folium.Icon(color=color)).add_to(citation_map)
    else:
        for affiliation_name in affiliation_map:
            corresponding_entries = affiliation_map[affiliation_name]
            author_name_list = []
            location_valid = True
            for entry_idx in corresponding_entries:
                author_name, _, _, _, lat, lon, _, _, _, _  = coordinates_and_info[entry_idx]
                if pd.isna(lat) or pd.isna(lon) or lat == '' or lon == '':
                    location_valid = False
                author_name_list.append(author_name)
            if location_valid:
                folium.Marker([lat, lon], popup='%s (%s)' % (affiliation_name, ' & '.join(author_name_list))).add_to(citation_map)
    return citation_map

def count_citation_stats(coordinates_and_info: List[Tuple[str]]) -> List[int]:
    '''
    Count the number of citing authors, affiliations and countries.
    '''
    unique_author_list, unique_affiliation_list, unique_country_list = set(), set(), set()
    for (author_name, _, _, affiliation_name, _, _, _, _, _, country) in coordinates_and_info:
        if affiliation_name == NO_AUTHOR_FOUND_STR:
            continue
        unique_author_list.add(author_name)
        unique_affiliation_list.add(affiliation_name)
        if country is not None and not (isinstance(country, float) and pd.isna(country)):
            unique_country_list.add(normalize_country(country))
    num_authors, num_affiliations, num_countries = \
        len(unique_author_list), len(unique_affiliation_list), len(unique_country_list)
    return num_authors, num_affiliations, num_countries


def __country_aware_comma_split(string_list: List[str]) -> List[str]:
    comma_split_list = []

    for part in string_list:
        # Split the strings by comma.
        # NOTE: The non-English comma is entered intentionally.
        sub_parts = [sub_part.strip() for sub_part in re.split(r'[,，]', part)]
        sub_parts_iter = iter(sub_parts)

        # Merge the split strings if the latter component is a country name.
        for sub_part in sub_parts_iter:
            if __iscountry(sub_part):
                continue  # Skip country names if they appear as the first sub_part.
            next_part = next(sub_parts_iter, None)
            if __iscountry(next_part):
                comma_split_list.append(f"{sub_part}, {next_part}")
            else:
                comma_split_list.append(sub_part)
                if next_part:
                    comma_split_list.append(next_part)
    return comma_split_list

def __iscountry(string: str) -> bool:
    if not string:
        return False
    try:
        pycountry.countries.lookup(string)
        return True
    except LookupError:
        return False

def __print_author_and_affiliation(author_paper_affiliation_tuple_list: List[Tuple[str]]) -> None:
    __author_affiliation_tuple_list = []
    for author_name, _, _, affiliation_name in sorted(author_paper_affiliation_tuple_list):
        if author_name == NO_AUTHOR_FOUND_STR:
            continue
        __author_affiliation_tuple_list.append((author_name, affiliation_name))

    # Take unique tuples.
    __author_affiliation_tuple_list = list(set(__author_affiliation_tuple_list))
    for author_name, affiliation_name in sorted(__author_affiliation_tuple_list):
        print('Author: %s. Affiliation: %s.' % (author_name, affiliation_name))
    print('')
    return


def _cache_path(cache_folder: Optional[str], scholar_id: Optional[str], filename: str) -> Optional[str]:
    if cache_folder is None or scholar_id is None:
        return None
    return os.path.join(cache_folder, scholar_id, filename)

def save_cache(data: Any, fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        pickle.dump(data, fd)

def load_cache(fpath: str) -> Any:
    with open(fpath, "rb") as fd:
        return pickle.load(fd)

def run_openalex_pipeline(scholar_id: str,
                          openalex_api_key: Optional[str] = None,
                          openalex_mailto: Optional[str] = None,
                          cache_folder: Optional[str] = 'cache') -> List[Tuple[str]]:
    '''
    Build the coordinate list using OpenAlex instead of Google Scholar's
    (CAPTCHA-protected) citing-paper search.

    The author's publication titles are read from their Google Scholar profile
    (that endpoint is not blocked), matched to OpenAlex works, and the citing
    authors + institutions (with built-in coordinates) are fetched from OpenAlex.
    Every stage is checkpointed so an interrupted run resumes from cache.
    '''
    client = OpenAlexClient(api_key=openalex_api_key, mailto=openalex_mailto)

    # ---- Step 1: the author's publication titles (via Google Scholar profile). ----
    pubs_cache = _cache_path(cache_folder, scholar_id, 'publications_and_cites.pkl')
    if pubs_cache is not None and os.path.exists(pubs_cache):
        author_name, cites_list = load_cache(pubs_cache)
    else:
        author_name, cites_list = get_publications_and_cites(scholar_id)
        if pubs_cache is not None:
            save_cache((author_name, cites_list), pubs_cache)
    titles = list(dict.fromkeys(title for _, title in cites_list if title))
    print('Profile: %s. %d cited publication titles to match on OpenAlex.\n' % (author_name, len(titles)))

    # ---- Step 2: match titles to OpenAlex works (verifying the author). ----
    works_cache = _cache_path(cache_folder, scholar_id, 'openalex_works.pkl')
    if works_cache is not None and os.path.exists(works_cache):
        works = load_cache(works_cache)
    else:
        works = match_works_by_titles(client, titles, author_name_hint=author_name)
        if works_cache is not None:
            save_cache(works, works_cache)
    print('Matched %d/%d publications to OpenAlex works.\n' % (len(works), len(titles)))

    # ---- Step 3: citing records per work (incremental checkpoint). ----
    citing_cache = _cache_path(cache_folder, scholar_id, 'openalex_citing_by_work.pkl')
    citing_by_work = {}
    if citing_cache is not None and os.path.exists(citing_cache):
        citing_by_work = load_cache(citing_cache)
    for work_id, cited_title in tqdm(works, desc='Fetching citing works from OpenAlex', total=len(works)):
        if work_id in citing_by_work:
            continue
        citing_by_work[work_id] = get_citing_records(client, work_id, cited_title)
        if citing_cache is not None:
            save_cache(citing_by_work, citing_cache)

    all_records = []
    for recs in citing_by_work.values():
        all_records.extend(recs)
    print('\n%d citing (author, institution) records collected.\n' % len(all_records))

    # ---- Step 4: resolve institutions to coordinates (cached). ----
    geo_cache_path = _cache_path(cache_folder, scholar_id, 'openalex_institution_geo.pkl')
    institution_geo = {}
    if geo_cache_path is not None and os.path.exists(geo_cache_path):
        institution_geo = load_cache(geo_cache_path)
    needed = [inst_id for (_, _, _, inst_id, _, _) in all_records
              if inst_id and inst_id not in institution_geo]
    if needed:
        institution_geo.update(resolve_institution_geo(client, needed))
        if geo_cache_path is not None:
            save_cache(institution_geo, geo_cache_path)

    # ---- Step 5: build the 10-tuple coordinate list. ----
    coordinates_and_info = []
    for author_name_c, citing_title, cited_title, inst_id, inst_name, _ in all_records:
        geo = institution_geo.get(inst_id) if inst_id else None
        if geo and geo.get('latitude') is not None and geo.get('longitude') is not None:
            coordinates_and_info.append((
                author_name_c, citing_title, cited_title, geo.get('display_name') or inst_name or '',
                geo['latitude'], geo['longitude'],
                '', geo.get('city') or '', geo.get('region') or '',
                normalize_country(geo.get('country'))))
        else:
            # No institution or no coordinates: keep the record (uncounted on the
            # map) with an empty location so it still appears in the CSV.
            coordinates_and_info.append((
                author_name_c, citing_title, cited_title, inst_name or '',
                '', '', '', '', '', ''))

    coordinates_and_info = sorted(list(set(coordinates_and_info)))
    return coordinates_and_info


def generate_citation_map(scholar_id: str,
                          output_path: str = 'citation_map.html',
                          csv_output_path: str = 'citation_info.csv',
                          parse_csv: bool = False,
                          cache_folder: str = 'cache',
                          affiliation_conservative: bool = False,
                          num_processes: int = 16,
                          use_proxy: bool = False,
                          use_browser: bool = False,
                          request_interval: float = 1.0,
                          source: str = 'scholar',
                          openalex_api_key: Optional[str] = None,
                          openalex_mailto: Optional[str] = None,
                          pin_colorful: bool = True,
                          print_citing_affiliations: bool = True):
    '''
    Google Scholar Citation World Map.

    Parameters
    ----
    scholar_id: str
        Your Google Scholar ID.
    output_path: str
        (default is 'citation_map.html')
        The path to the output HTML file.
    csv_output_path: str
        (default is 'citation_info.csv')
        The path to the output csv file.
    parse_csv: bool
        (default is False)
        If True, will directly jump to Step 5.2, using the information loaded from the csv.
    cache_folder: str
        (default is 'cache')
        The folder to save intermediate results. Scraping is checkpointed incrementally
        (per citation page and per citing author), so if the run is interrupted (e.g. by a
        CAPTCHA), simply re-run to resume from where it stopped.
        Set to None if you do not want caching.
    affiliation_conservative: bool
        (default is False)
        If true, we will use a more conservative approach to identify affiliations
        (only the verified/linked organization). Higher precision, lower recall.
        If false, we will use the self-reported affiliation string. Higher recall.
    num_processes: int
        (default is 16)
        Kept for backward compatibility. Scraping is now performed sequentially to
        minimize the risk of being blocked by Google Scholar, so this is only used
        where safe.
    use_proxy: bool
        (default is False)
        If true, we will route Scholar requests through the proxy defined by the
        HTTP_PROXY / HTTPS_PROXY environment variables.
        It is necessary for some environments to avoid blocks, but it usually makes things slower.
    request_interval: float
        (default is 1.0)
        Minimum number of seconds between Google Scholar requests. Increase this if
        you are being blocked; decrease it (carefully) to go faster.
    pin_colorful: bool
        (default is True)
        If true, the location pins will have a variety of colors.
        Otherwise, it will only have one color.
    print_citing_affiliations: bool
        (default is True)
        If true, print the list of citing affiliations (affiliations of citing authors).
    '''

    if not parse_csv and source == 'openalex':
        # OpenAlex data source: free API, no CAPTCHA, institutions carry coordinates.
        coordinates_and_info = run_openalex_pipeline(
            scholar_id,
            openalex_api_key=openalex_api_key,
            openalex_mailto=openalex_mailto,
            cache_folder=cache_folder)
        export_dict_to_csv(coordinates_and_info, csv_output_path)
        print('\nCitation information exported to %s.' % csv_output_path)

    elif not parse_csv:

        # Configure the shared HTTP fetcher (request interval + optional proxy).
        proxies = None
        if use_proxy:
            http_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if http_proxy:
                proxies = {'http': http_proxy, 'https': http_proxy}
                print('Using proxy: %s' % http_proxy)
            else:
                print('[Warning] use_proxy=True but no HTTP_PROXY/HTTPS_PROXY env var found.')
        configure_fetcher(min_interval=request_interval, proxies=proxies, use_browser=use_browser)

        # NOTE: Step 1 & 2. Find all publications and citing authors (checkpointed).
        all_citing_author_paper_tuple_list = find_all_citing_authors(
            scholar_id=scholar_id, cache_folder=cache_folder)
        print('A total of %d citing (author, paper) records found.\n' % len(all_citing_author_paper_tuple_list))

        # NOTE: Step 3. Find all citing affiliations (checkpointed).
        print('Identifying affiliations using the %s approach.' % (
            'conservative' if affiliation_conservative else 'aggressive'))
        author_paper_affiliation_tuple_list = find_all_citing_affiliations(
            all_citing_author_paper_tuple_list,
            affiliation_conservative=affiliation_conservative,
            scholar_id=scholar_id,
            cache_folder=cache_folder)
        print('\nA total of %d citing affiliations recorded.\n' % len(author_paper_affiliation_tuple_list))
        author_paper_affiliation_tuple_list = list(set(author_paper_affiliation_tuple_list))

        # NOTE: Step 3b. Clean the affiliation strings (only used in the aggressive approach).
        if print_citing_affiliations:
            if affiliation_conservative:
                print('Taking the conservative approach. Will not need to clean the affiliation names.')
            else:
                print('Taking the aggressive approach. Cleaning the affiliation names.')
            __print_author_and_affiliation(author_paper_affiliation_tuple_list)
        if not affiliation_conservative:
            cleaned_author_paper_affiliation_tuple_list = clean_affiliation_names(author_paper_affiliation_tuple_list)
            if print_citing_affiliations:
                print('List of all citing authors and affiliations after cleaning:\n')
                __print_author_and_affiliation(cleaned_author_paper_affiliation_tuple_list)
            # Use the merged set to maximize coverage.
            author_paper_affiliation_tuple_list += cleaned_author_paper_affiliation_tuple_list
            author_paper_affiliation_tuple_list = list(set(author_paper_affiliation_tuple_list))

        # NOTE: Step 4. Convert affiliations in plain text to Geocode.
        coordinates_and_info = affiliation_text_to_geocode(
            author_paper_affiliation_tuple_list,
            scholar_id=scholar_id,
            cache_folder=cache_folder)
        # Take unique tuples.
        coordinates_and_info = sorted(list(set(coordinates_and_info)))

        # NOTE: Step 5.1. Export csv file recording citation information.
        export_dict_to_csv(coordinates_and_info, csv_output_path)
        print('\nCitation information exported to %s.' % csv_output_path)

    else:
        print('\nDirectly parsing the csv. Skipping all previous steps.')
        assert os.path.isfile(csv_output_path), '`csv_output_path` is not a file.'
        coordinates_and_info = read_csv_to_dict(csv_output_path)
        print('\nCitation information loaded from %s.' % csv_output_path)

    # NOTE: Step 5.2. Create the citation world map.
    citation_map = create_map(coordinates_and_info, pin_colorful=pin_colorful)
    citation_map.save(output_path)
    print('\nHTML map created and saved at %s.\n' % output_path)

    num_authors, num_affiliations, num_countries = count_citation_stats(coordinates_and_info)
    print('\nYou have been cited by %s researchers from %s affiliations and %s countries.\n' % (
        num_authors, num_affiliations, num_countries))
    return


if __name__ == '__main__':
    # Replace this with your Google Scholar ID.
    scholar_id = '3rDjnykAAAAJ'
    generate_citation_map(scholar_id,
                          output_path='citation_map.html',
                          csv_output_path='citation_info.csv',
                          parse_csv=False,
                          cache_folder='cache',
                          affiliation_conservative=False,
                          num_processes=16,
                          use_proxy=False,
                          request_interval=1.0,
                          pin_colorful=True,
                          print_citing_affiliations=True)
