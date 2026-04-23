import os
import argparse
import pyproj
import time
import requests
import pandas as pd
import hashlib
import hmac
import base64
import urllib.parse as urlparse

from tqdm import tqdm
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')

def bng_to_wgs84(easting, northing):
    # Define the coordinate systems
    bng = pyproj.Proj('epsg:27700')  # BNG
    wgs84 = pyproj.Proj('epsg:4326')  # WGS84

    # Convert the coordinates
    lon, lat = pyproj.transform(bng, wgs84, easting, northing)

    return lon, lat

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

def get_base_url(api_key, lon, lat, zoom=17, size='640x640', maptype='satellite'):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        'center': f'{lon},{lat}',
        'zoom': zoom,
        'size': size,
        'maptype': maptype,
        'key': api_key,
    }
    response = requests.Request('GET', base_url, params=params).prepare()
    url = response.url

    return url

def save_image(image, eid, zoom_level, output_folder, is_crop=False):
    image_path = os.path.join(output_folder, f"bev_image_eid-{int(eid)}_zoom-{zoom_level}.png")
    image = Image.open(BytesIO(image))

    if is_crop:
        image = image.crop((20, 20, 620, 620))

    image.save(image_path, format='PNG')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BEV satellite images using Google Maps Static API.")
    parser.add_argument("--coordinates_file", type=str, required=True,
                        help="Path to TSV file with columns: eid, home_location_east_coordinate, home_location_north_coordinate")
    parser.add_argument("--api_key_file", type=str, required=True,
                        help="Path to file containing Google Maps API key")
    parser.add_argument("--signature_key_file", type=str, required=True,
                        help="Path to file containing URL signing secret")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save downloaded images")
    parser.add_argument("--zoom_levels", type=str, default="15,17",
                        help="Comma-separated zoom levels (default: 15,17)")
    parser.add_argument("--crop", action="store_true", default=True,
                        help="Crop images to remove Google watermark (default: True)")
    args = parser.parse_args()

    with open(args.api_key_file, "r") as f:
        api_key = f.read().strip()

    with open(args.signature_key_file, "r") as f:
        signature = f.read().strip()

    df = pd.read_csv(args.coordinates_file, delimiter="\t")
    df = df.loc[:, ["eid", "home_location_east_coordinate", "home_location_north_coordinate"]]

    zoom_levels = [int(z) for z in args.zoom_levels.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        eid = int(row['eid'])
        easting = row['home_location_east_coordinate']
        northing = row['home_location_north_coordinate']

        lon, lat = bng_to_wgs84(easting, northing)

        for zoom_level in zoom_levels:
            out_path = os.path.join(args.output_dir, f"bev_image_eid-{eid}_zoom-{zoom_level}.png")
            if not os.path.exists(out_path):
                base_url = get_base_url(api_key, lon, lat, zoom=zoom_level)
                signed_url = sign_url(base_url, signature)

                response = requests.get(signed_url)
                image = response.content
                save_image(image, eid, zoom_level, output_folder=args.output_dir, is_crop=args.crop)
                time.sleep(0.01)
