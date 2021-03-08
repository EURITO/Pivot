from nesta.packages.geo_utils.country_iso_code import country_iso_code_to_name

from indicators.core.config import INDICATORS
from indicators.core.core_utils import flatten
from indicators.core.nuts_utils import get_nuts_info_lookup


import boto3
import pandas as pd
from collections import defaultdict
from pathlib import Path


def make_indicator_description(indicator_name, topic_module):
    """
    Retrieve and format the verbose indicator name for this dataset

    Args:
        indicator_name (str): An indicator name, as specified both in the
                              `indicators.yaml` config and in the
                              `indicators.two.generate_indicators` function.

        topic_module (module): Nominally one of `arxiv_topics`,
                               `nih_topics`, `cordis_topics`, which all
                               have a consistent API.
    Returns:
        verbose_name (str): The verbose indicator name for this dataset
    """
    # Retrieve one of "articles" or "projects" from the module's config
    entity_type = topic_module.model_config["metadata"]["entity_type"]
    unformatted_name = INDICATORS["verbose_indicator_names"][indicator_name]
    return unformatted_name.format(entity_type)


def make_ctry_metadata(ctry_code):
    """
    Prepare metadata for this country code, including the standard
    filename for this country code. The code could be ISO (prefixed with
    "iso_") or NUTS.
    """
    nuts_lookup = get_nuts_info_lookup()  # NB: lru_cached
    nuts_level = len(ctry_code) - 1
    is_iso_code = ctry_code.startswith("iso_")  # i.e. is not NUTS code
    # Strip "iso_" if ISO, else use the bare NUTS code
    ctry_code = ctry_code[4:] if is_iso_code else ctry_code
    # Create the metadata object
    ctry_metadata = {
        "ctry_code": ctry_code,
        # Define level = 0 for ISOs, else work it out from code length
        "level": 0 if is_iso_code else len(ctry_code) - 1,
        # Get the name for this code
        "name": (
            country_iso_code_to_name(ctry_code, iso2=True)  # not iso3
            if is_iso_code
            else nuts_lookup[ctry_code]["nuts_name"]
        ),
        # Define a standard file name depending on whether ISO or not
        "filename": "by-country.csv" if is_iso_code else f"nuts-{nuts_level}.csv",
    }
    return ctry_metadata


def prepare_file_data(indicators):
    """
    Flatten indicator data from the form [dataset][country][indicator][topic] and enrich with country-level metadata (i.e. code, level, name).

    Returns:
        file_data (dict of list): Filepaths pointing to a list of curated data.
    """
    # Mapping of filepaths pointing to a list of curated data
    all_file_data = defaultdict(list)
    # Flatten fields from the form [dataset][country][indicator][topic]
    for ds_name, ctry_code, indc_name, topic_name, indc_value in flatten(indicators):
        # Ignore null rows
        if pd.isnull(indc_value) or indc_value == 0:
            continue
        # Prepare metadata
        indc_desc = make_indicator_description(indc_name, ds_name)
        ctry_metadata = make_ctry_metadata(ctry_code)
        filename = ctry_metadata.pop("filename")
        filepath = f"{ds_name}/{topic_name}/{filename}"
        # Add this row to the output
        file_data = all_file_data[filepath]  # create if does not exist
        file_data.append(
            {
                "indicator_name": indc_name,
                "indicator_value": indc_value,
                "indicator_description": indc_desc,
                **ctry_metadata,
            }
        )
    return all_file_data


def sort_and_filter_data(file_data):
    """Take file data, i.e. output from `prepare_file_data`, and
    convert to DataFrame (for easy to_csv saving), and sort fields"""
    sorted_file_data = {}
    for path, data in file_data.items():
        df = pd.DataFrame(data)
        df = df.sort_values(by=INDICATORS["variable_order"])
        df = df.reset_index(drop=True)
        df = df.dropna(axis=0)
        if len(df) == 0:
            continue
        sorted_file_data[path] = df
    return sorted_file_data


def save_and_upload(sorted_file_data):
    """Save the data locally and to S3"""
    s3 = boto3.resource("s3")
    bucket_name = INDICATORS["bucket_name"]
    for path, data in sorted_file_data.items():
        # Create the local save path, if not already done so
        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        # Save the data locally
        data.to_csv(path, index=False)
        # Save to S3
        s3.Bucket(bucket_name).upload_file(path, path)


def _days_of_covid():
    """Number of days since Covid emergency declared"""
    covid_start = INDICATORS["covid_dates"]["from_date"]
    covid_end = INDICATORS["covid_dates"]["to_date"]
    diff = (pd.to_datetime(covid_end) - pd.to_datetime(covid_start)).days
    return diff + 1  # inclusive difference


def safe_divide(numerator, denominator):
    """
    In the case where there is no past activity, take 1 as an upper bound
    """
    denominator.loc[denominator == 0] = 1
    return numerator / denominator


def sort_save_and_upload(indicators):
    """
    Flatten indicators into the form [filename][data]
    before sorting columns and saving to disk and then S3.
    """
    # Flatten indicators into the form [filename][data]
    file_data = prepare_file_data(indicators)
    sorted_file_data = sort_and_filter_data(file_data)
    # Save to disk and S3
    save_and_upload(sorted_file_data)


# Convert to parameter for nicer interface
days_of_covid = _days_of_covid()
