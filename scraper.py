import musicbrainzngs
import requests
import json
import time
import hashlib
import urllib.parse
import logging
import pandas as pd
import os
import warnings
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

# Configuration
LASTFM_API_KEY = "YOUR_LASTFM_API_KEY"  # Replace with your key
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Setup MusicBrainz
# musicbrainzngs.set_useragent(
#     "Music4AllPipeline",
#     "1.0",
#     "contact@example.com"  # Replace with your email
# )

warnings.filterwarnings("ignore", module="musicbrainzngs")
logging.getLogger('musicbrainzngs').setLevel(logging.ERROR)
logging.getLogger('musicbrainzngs.mbxml').setLevel(logging.ERROR)
logging.getLogger('musicbrainzngs.util').setLevel(logging.ERROR)


# Setup logging with UTF-8 encoding for file, ASCII for console
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace Unicode characters with ASCII equivalents
            msg = msg.replace('✓', '[OK]').replace('↓', '>>').replace('✗', '[X]')
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []

# File handler with UTF-8
file_handler = logging.FileHandler('music4all_pipeline.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(file_handler)

# Console handler with safe encoding
console_handler = SafeStreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console_handler)


class Music4AllCompletePipeline:
    """Complete pipeline for Music4All dataset creation."""

    def __init__(self, csv_path='id_information.csv'):
        self.csv_path = csv_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Music4AllPipeline/1.0'
        })

        # Data storage
        self.api_data = {}
        self.unique_artist_mbids = OrderedDict()
        self.unique_album_mbids = OrderedDict()

        # Statistics
        self.stats = {
            'tracks_processed': 0,
            'artists_found': 0,
            'albums_found': 0,
            'artist_images': 0,
            'album_images': 0,
            'lastfm_album_hits': 0,
            'lastfm_album_misses': 0
        }

        # Create directories
        os.makedirs('artists', exist_ok=True)
        os.makedirs('albums', exist_ok=True)

        # Load existing data
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing api_data.json if available."""
        if os.path.exists('api_data.json'):
            try:
                with open('api_data.json', 'r', encoding='utf-8') as f:
                    self.api_data = json.load(f)

                # Count existing entries
                for mbid, data in self.api_data.items():
                    if 'artist_info' in data:
                        self.unique_artist_mbids[mbid] = {
                            'track_id': 'existing',
                            'image_count': data.get('image_count', 0)
                        }
                    elif 'album_info' in data:
                        self.unique_album_mbids[mbid] = {
                            'track_id': 'existing',
                            'image_count': data.get('image_count', 0)
                        }

                logging.info(f"[OK] Loaded {len(self.api_data)} existing entries ({len(self.unique_artist_mbids)} artists, {len(self.unique_album_mbids)} albums)")
            except Exception as e:
                logging.error(f"Error loading existing data: {e}")

    def load_track_data(self):
        """Load track data from CSV."""
        logging.info(f"Loading track data from {self.csv_path}...")
        try:
            try:
                self.track_data = pd.read_csv(self.csv_path, sep='\t', encoding='utf-8')
            except:
                self.track_data = pd.read_csv(self.csv_path, encoding='utf-8')

            if len(self.track_data.columns) == 4:
                self.track_data.columns = ['id', 'artist', 'song', 'album_name']

            logging.info(f"[OK] Loaded {len(self.track_data)} tracks")

            # Show sample data
            logging.info("Sample tracks:")
            for i in range(min(3, len(self.track_data))):
                row = self.track_data.iloc[i]
                logging.info(f"  {i + 1}. '{row['song']}' by {row['artist']} (Album: {row.get('album_name', 'N/A')})")

            return True
        except Exception as e:
            logging.error(f"[X] Error loading CSV: {e}")
            return False

    def get_track_mbids_from_lastfm(self, artist: str, track: str, album: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Get MBIDs from Last.fm track.getInfo."""
        try:
            params = {
                'method': 'track.getInfo',
                'api_key': LASTFM_API_KEY,
                'artist': artist,
                'track': track,
                'format': 'json'
            }

            response = self.session.get(LASTFM_API_URL, params=params, timeout=10)
            if response.status_code == 404:
                return None, None

            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                return None, None

            if 'track' not in data:
                return None, None

            track_data = data['track']
            artist_mbid = track_data.get('artist', {}).get('mbid', '').strip() or None
            album_mbid = track_data.get('album', {}).get('mbid', '').strip() or None

            return artist_mbid, album_mbid

        except Exception as e:
            return None, None

    def search_artist_mbid_musicbrainz(self, artist_name: str) -> Optional[str]:
        """Search for artist MBID using MusicBrainz."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = musicbrainzngs.search_artists(artist=artist_name, limit=1)

            if result and 'artist-list' in result and result['artist-list']:
                best_match = result['artist-list'][0]
                score = best_match.get('ext:score', '100')
                score = int(score) if isinstance(score, str) else score

                if score >= 90:
                    mbid = best_match['id']
                    return mbid
            return None
        except Exception as e:
            return None

    def get_artist_data_from_musicbrainz(self, artist_mbid: str) -> Optional[Dict]:
        """Get artist data from MusicBrainz."""
        try:
            logging.info(f">> Fetching artist from MusicBrainz: {artist_mbid[:8]}...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = musicbrainzngs.get_artist_by_id(
                    artist_mbid,
                    includes=["url-rels", "aliases", "tags", "ratings"]
                )

            time.sleep(1.5)

            artist_data = result.get('artist', {})
            artist_name = artist_data.get('name', 'Unknown')

            # Extract genres from tags
            genres = {}
            if 'tag-list' in artist_data:
                for tag in artist_data['tag-list']:
                    genres[tag['name']] = int(tag.get('count', 0))

            # Find URLs
            wikidata_url = None
            wikipedia_url = None
            image_urls = []

            for rel in artist_data.get('url-relation-list', []):
                rel_type = rel.get('type', '')
                target = rel.get('target', '')

                if rel_type == 'wikidata':
                    wikidata_url = target
                elif rel_type == 'wikipedia':
                    wikipedia_url = target
                elif rel_type == 'image':
                    image_urls.append(target)

            # Get Wikipedia summary
            wiki_summary = None
            if wikidata_url:
                wiki_summary = self.get_wikipedia_summary(wikidata_url)
            elif wikipedia_url:
                wiki_summary = self.get_wikipedia_summary_from_url(wikipedia_url)

            if wiki_summary:
                artist_data['wiki'] = {'summary': wiki_summary}

            # Get image
            artist_image_url = None
            if wikidata_url:
                artist_image_url = self.get_artist_image_from_wikidata(wikidata_url)
            if not artist_image_url and image_urls:
                artist_image_url = image_urls[0]

            # Download image
            image_count = 0
            if artist_image_url:
                image_count = self.download_artist_image(artist_mbid, artist_image_url)
                if image_count:
                    self.stats['artist_images'] += 1

            self.stats['artists_found'] += 1
            logging.info(f"  [OK] Artist: {artist_name} (genres: {len(genres)}, image: {'yes' if image_count else 'no'})")

            return {
                'artist_info': {'artist': artist_data},
                'artist_image_url': artist_image_url,
                'mbid': artist_mbid,
                'genres': genres,
                'image_count': image_count
            }

        except Exception as e:
            logging.error(f"  [X] Error fetching artist: {e}")
            return None

    def get_album_data_from_lastfm(self, album_mbid: str) -> Optional[Dict]:
        """Get album data from Last.fm."""
        try:
            logging.info(f">> Fetching album from Last.fm: {album_mbid[:8]}...")

            # Get basic structure from MusicBrainz
            album_data = {}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mb_result = musicbrainzngs.get_release_by_id(
                        album_mbid,
                        includes=["release-groups", "recordings", "artist-credits"]
                    )
                album_data = mb_result.get('release', {})
            except:
                album_data = {'mbid': album_mbid}

            # Get Last.fm data
            params = {
                'method': 'album.getInfo',
                'mbid': album_mbid,
                'api_key': LASTFM_API_KEY,
                'format': 'json'
            }

            response = self.session.get(LASTFM_API_URL, params=params, timeout=10)

            if response.status_code == 404:
                logging.info(f"  [X] Album not found on Last.fm")
                self.stats['lastfm_album_misses'] += 1
                return None

            response.raise_for_status()
            lastfm_data = response.json()

            if 'error' in lastfm_data:
                logging.info(f"  [X] Last.fm error: {lastfm_data.get('message', 'Unknown')}")
                self.stats['lastfm_album_misses'] += 1
                return None

            genres = {}
            cover_url = None
            album_name = "Unknown"

            if 'album' in lastfm_data:
                album_info = lastfm_data['album']
                album_name = album_info.get('name', 'Unknown')

                # Extract genres
                if 'tags' in album_info and 'tag' in album_info['tags']:
                    for tag in album_info['tags']['tag']:
                        genres[tag['name']] = 100

                # Get cover
                if 'image' in album_info:
                    for img in album_info['image']:
                        if img.get('size') == 'extralarge' and img.get('#text'):
                            cover_url = img['#text']
                            break

                album_data['listeners'] = album_info.get('listeners', 0)
                album_data['playcount'] = album_info.get('playcount', 0)

                self.stats['lastfm_album_hits'] += 1
            else:
                self.stats['lastfm_album_misses'] += 1
                return None

            # Download cover
            image_count = 0
            if cover_url:
                image_count = self.download_album_cover(album_mbid, cover_url)
                if image_count:
                    self.stats['album_images'] += 1

            self.stats['albums_found'] += 1
            logging.info(f"  [OK] Album: {album_name} (genres: {len(genres)}, cover: {'yes' if image_count else 'no'})")

            return {
                'album_info': {'album': album_data},
                'album_cover_url': cover_url,
                'mbid': album_mbid,
                'genres': genres,
                'image_count': image_count
            }

        except Exception as e:
            logging.error(f"  [X] Error fetching album: {e}")
            self.stats['lastfm_album_misses'] += 1
            return None

    def get_wikipedia_summary(self, wikidata_url: str) -> Optional[str]:
        """Get Wikipedia summary from Wikidata URL."""
        try:
            wikidata_id = wikidata_url.split('/')[-1]

            params = {
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "props": "sitelinks"
            }

            response = self.session.get(WIKIDATA_API_URL, params=params, timeout=10)
            data = response.json()

            entity = data.get('entities', {}).get(wikidata_id, {})
            sitelinks = entity.get('sitelinks', {})

            if 'enwiki' in sitelinks:
                wikipedia_title = sitelinks['enwiki']['title']

                wiki_params = {
                    "action": "query",
                    "titles": wikipedia_title,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "format": "json"
                }

                wiki_response = self.session.get(WIKIPEDIA_API_URL, params=wiki_params, timeout=10)
                wiki_data = wiki_response.json()

                pages = wiki_data.get('query', {}).get('pages', {})
                if pages:
                    page_id = next(iter(pages))
                    return pages[page_id].get('extract', '')

            return None
        except:
            return None

    def get_wikipedia_summary_from_url(self, wikipedia_url: str) -> Optional[str]:
        """Get Wikipedia summary from URL."""
        try:
            title = wikipedia_url.split('/')[-1]
            title = urllib.parse.unquote(title)

            params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json"
            }

            response = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if pages:
                page_id = next(iter(pages))
                return pages[page_id].get('extract', '')

            return None
        except:
            return None

    def get_artist_image_from_wikidata(self, wikidata_url: str) -> Optional[str]:
        """Get artist image URL from Wikidata."""
        try:
            wikidata_id = wikidata_url.split('/')[-1]

            params = {
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "props": "claims"
            }

            response = self.session.get(WIKIDATA_API_URL, params=params, timeout=10)
            data = response.json()

            entity = data.get('entities', {}).get(wikidata_id, {})
            claims = entity.get('claims', {})

            if 'P18' in claims:
                image_claim = claims['P18'][0]
                if 'mainsnak' in image_claim and 'datavalue' in image_claim['mainsnak']:
                    filename = image_claim['mainsnak']['datavalue']['value']

                    processed_filename = filename.replace(" ", "_")
                    m = hashlib.md5()
                    m.update(processed_filename.encode('utf-8'))
                    hashed = m.hexdigest()

                    base_url = "https://upload.wikimedia.org/wikipedia/commons/"
                    encoded_filename = urllib.parse.quote(processed_filename)

                    return f"{base_url}{hashed[0]}/{hashed[0:2]}/{encoded_filename}"

            return None
        except:
            return None

    def download_artist_image(self, mbid: str, image_url: str) -> int:
        """Download artist image."""
        try:
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()

            image_path = f"artists/{mbid}_0.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return 1
        except:
            return 0

    def download_album_cover(self, mbid: str, cover_url: str) -> int:
        """Download album cover."""
        try:
            response = self.session.get(cover_url, timeout=30)
            response.raise_for_status()

            image_path = f"albums/{mbid}_0.jpg"
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return 1
        except:
            return 0

    def process_all_tracks(self):
        """Process all tracks."""
        logging.info("\n" + "=" * 60)
        logging.info("Starting pipeline processing...")
        logging.info("=" * 60 + "\n")

        total = len(self.track_data)
        processed = 0

        for idx, row in self.track_data.iterrows():
            track_id = str(row['id']).strip()
            artist = str(row['artist']).strip()
            song = str(row['song']).strip()
            album = str(row.get('album_name', '')).strip()
            album = album if album and album != 'nan' else None

            # Get MBIDs from Last.fm
            artist_mbid, album_mbid = self.get_track_mbids_from_lastfm(artist, song, album)

            # If no artist MBID, try MusicBrainz
            if not artist_mbid:
                artist_mbid = self.search_artist_mbid_musicbrainz(artist)

            # Process artist
            if artist_mbid and artist_mbid not in self.unique_artist_mbids and artist_mbid not in self.api_data:
                artist_data = self.get_artist_data_from_musicbrainz(artist_mbid)
                if artist_data:
                    self.api_data[artist_mbid] = artist_data
                    self.unique_artist_mbids[artist_mbid] = {
                        'track_id': track_id,
                        'image_count': artist_data.get('image_count', 0)
                    }

            # Process album
            if album_mbid and album_mbid not in self.unique_album_mbids and album_mbid not in self.api_data:
                album_data = self.get_album_data_from_lastfm(album_mbid)
                if album_data:
                    self.api_data[album_mbid] = album_data
                    self.unique_album_mbids[album_mbid] = {
                        'track_id': track_id,
                        'image_count': album_data.get('image_count', 0)
                    }

            processed += 1
            self.stats['tracks_processed'] = processed

            # Progress update
            if processed % 50 == 0:
                pct = (processed / total) * 100
                new_artists = len([k for k, v in self.unique_artist_mbids.items() if v['track_id'] != 'existing'])
                new_albums = len([k for k, v in self.unique_album_mbids.items() if v['track_id'] != 'existing'])

                logging.info(f"\nProgress: {processed}/{total} tracks ({pct:.1f}%)")
                logging.info(f"  New artists: {new_artists}, New albums: {new_albums}")
                logging.info(f"  Images: {self.stats['artist_images']} artists, {self.stats['album_images']} albums")
                logging.info(f"  Last.fm albums: {self.stats['lastfm_album_hits']} hits, {self.stats['lastfm_album_misses']} misses")

                self.save_api_data()

            # Rate limiting
            time.sleep(0.3)

        logging.info("\n" + "=" * 60)
        logging.info("Processing complete!")
        logging.info("=" * 60)

    def save_api_data(self):
        """Save api_data.json."""
        with open('api_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.api_data, f, indent=4, ensure_ascii=False)

    def save_mappings(self):
        """Save mapping CSV files."""
        new_artists = {k: v for k, v in self.unique_artist_mbids.items() if v['track_id'] != 'existing'}
        new_albums = {k: v for k, v in self.unique_album_mbids.items() if v['track_id'] != 'existing'}

        with open('artist_id_to_mbid_mapping.csv', 'w', encoding='utf-8') as f:
            f.write("id,mbid,image_count\n")
            for mbid, data in new_artists.items():
                f.write(f"{data['track_id']},{mbid},{data['image_count']}\n")

        with open('album_id_to_mbid_mapping.csv', 'w', encoding='utf-8') as f:
            f.write("id,mbid,image_count\n")
            for mbid, data in new_albums.items():
                f.write(f"{data['track_id']},{mbid},{data['image_count']}\n")

        logging.info(f"[OK] Saved mappings: {len(new_artists)} new artists, {len(new_albums)} new albums")


def main():
    if LASTFM_API_KEY == "YOUR_LASTFM_API_KEY":
        logging.error("Please set your Last.fm API key!")
        return

    pipeline = Music4AllCompletePipeline(os.path.join('Music4All', 'id_information.csv'))

    if not pipeline.load_track_data():
        return

    pipeline.process_all_tracks()
    pipeline.save_api_data()
    pipeline.save_mappings()

    logging.info("\n" + "=" * 60)
    logging.info("FINAL STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total entries: {len(pipeline.api_data)}")
    logging.info(f"Artists: {len(pipeline.unique_artist_mbids)} ({pipeline.stats['artist_images']} with images)")
    logging.info(f"Albums: {len(pipeline.unique_album_mbids)} ({pipeline.stats['album_images']} with covers)")
    logging.info(f"Last.fm album success rate: {pipeline.stats['lastfm_album_hits']}/{pipeline.stats['lastfm_album_hits'] + pipeline.stats['lastfm_album_misses']}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()