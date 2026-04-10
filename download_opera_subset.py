#!/usr/bin/env python3

import os
script_name = os.path.basename(__file__)
description = f"Download and subset Sentinel-1 displacement files from ASF for a specified region and time range.\n\n"
epi = f"Example usage: {script_name} --input_dir /Users/giacomo/onedrive/scratch/opera_download/Popcatepetl --polygon 'POLYGON((-98.7393 18.9444,-98.5146 18.9444,-98.5146 19.0774,-98.7393 19.0774,-98.7393 18.9444))' --flight_direction DESCENDING --start_date 20170101 --end_date 20170301"

from displacement_tools import download_disp_files, estimate_stack_size
import asf_search as asf
import netrc
import datetime
import argparse

def parse_polygon(polygon):
    latitude = []
    longitude = []
    pol = polygon.replace("POLYGON((", "").replace("))", "")
    for word in pol.split(','):
        if (float(word.split(' ')[1])) not in latitude:
            latitude.append(float(word.split(' ')[1]))
        if (float(word.split(' ')[0])) not in longitude:
            longitude.append(float(word.split(' ')[0]))
    longitude = [round(min(longitude),2), round(max(longitude),2)]
    latitude = [round(min(latitude),2), round(max(latitude),2)]
    region = [longitude[0], longitude[1], latitude[0], latitude[1]]
    return region

def main():
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter, epilog=epi)
    parser.add_argument('--input_dir', type=str, required=True, help='Output directory for downloaded files')
    parser.add_argument('--polygon', type=str, required=True, help='Polygon string in WKT format (e.g., "POLYGON((lon1 lat1, lon2 lat2, ...))")')
    parser.add_argument('--flight_direction', type=str, choices=['ASCENDING', 'DESCENDING'], required=True, help='Flight direction (ASCENDING or DESCENDING)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYYMMDD format')
    args = parser.parse_args()

    host = 'urs.earthdata.nasa.gov'
    parsed = netrc.netrc().authenticators(host)
    if not parsed:
        raise ValueError(f'No authentication credentials found in ~/.netrc for {host}')
    username = parsed[0]
    password = parsed[2]

    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.DISP_S1,
        start=datetime.datetime.strptime(args.start_date, '%Y%m%d').date(),
        end=datetime.datetime.strptime(args.end_date, '%Y%m%d').date(),
        intersectsWith=args.polygon,
        flightDirection=args.flight_direction,
        dataset=asf.DATASET.OPERA_S1,
    )

    bbox = parse_polygon(args.polygon)
    bbox_bounds = {
        "lon_min": bbox[0],
        "lon_max": bbox[1],
        "lat_min": bbox[2],
        "lat_max": bbox[3]
    }

    url = [getattr(r, 'properties')['url'] for r in results]
    file_size, subset_size, ratio, stack_gb, bbox = estimate_stack_size(url, bbox_bounds, username, password)

    print(f"Estimated total file size: {file_size:.2f} GB")
    print(f"Estimated subset size: {subset_size:.2f} GB")
    print(f"Estimated size reduction ratio: {ratio:.2f}")
    print(f"Estimated stack size after subsetting: {stack_gb:.2f} GB")
    print(f"BBOX for subsetting: {bbox}")

    download_disp_files(url, bbox, args.input_dir, username, password, 5)

if __name__ == "__main__":
    main()
