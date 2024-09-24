import os
from datetime import datetime
import json


def list_dir(folder, **kwargs):
    """
    Utility to retrieve files and/or folders from a given source folder, recursively or not,
    possibly filtered by certain specs
    :param folder: the folder to scan
    :param kwargs: options to filter-out elements of certain type or to keep only elements with given extension
    :return: a list of full paths of all the elements in folder
    """

    dir_only = kwargs.get("dir_only", False)
    files_only = kwargs.get("files_only", False)
    extension_filter = kwargs.get("extension_filter", "")
    assert not (dir_only and files_only and extension_filter), \
        "Arguments dir_only, files_only and extension_filter are mutually exclusive"

    apply_recursively = kwargs.get("apply_recursively", False)

    # scan folder recursively adding all the elements in the folder
    dir_contents = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        dir_contents.append(os.path.join(folder, name))
        if apply_recursively and os.path.isdir(path):
            dir_contents.extend(list_dir(folder=os.path.join(folder, name), **kwargs))

    # keep only the elements compliant to the filters specified
    if dir_only:
        dir_contents = [path for path in dir_contents if os.path.isdir(path)]
    elif files_only:
        dir_contents = [path for path in dir_contents if os.path.isfile(path)]
    elif extension_filter:
        dir_contents = [path for path in dir_contents if path.endswith(extension_filter)]

    return dir_contents


def decorate_str_with_date(value: str):
    """
    Add current datetime to a given string value

    :param value: a string
    :return: Value decorated with current datetime: value_%d_%m_%Y_%H_%M_%S
    """
    now = datetime.now()
    now = now.strftime("%d_%m_%Y_%H_%M_%S")
    return f"{value}_{now}"


def save_json(json_dict, json_file):
    """
    Save a dictionary as json.

    :param json_dict: The dictionary to save
    :param json_file: Json file destination
    :return:
    """
    with open(json_file, 'w') as outfile:
        json.dump(json_dict, outfile)


def load_json(json_file):
    """
    Load a dictionary from json.

    :param json_dict: The dictionary to save
    :param json_file: Json file destination
    :return:
    """
    with open(json_file, 'r', encoding="utf-8") as infile:
        return json.load(infile)


def join_and_create_if_missing(current_path, folder_name):
    """
    os.path.join like method that create the joined folder if it does
    not exist.

    :param current_path: An existing folder
    :param folder_name: the name of a folder to join with current path. Creates it if
    missing

    :return: the joined path
    """
    folder = os.path.join(current_path, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    return folder


def open_txt_file(fp):
    with open(fp, "r", encoding='utf-8') as f:
        txt = f.read()
    return txt