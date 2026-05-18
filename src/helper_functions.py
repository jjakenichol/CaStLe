"""
Helper functions for constructing result filenames used by experiment scripts.

Filenames embed hyperparameter values and an FDR method tag adjacent to a UUID
suffix so that results from different configurations can coexist in the same
output directory without collisions.
"""

import re


def add_fdr_to_filename(old_filename, fdr_method):
    """
    Insert an FDR method tag into a result filename.

    Expects the filename to contain a 32-character hex UUID. Prepends ``r_``
    and inserts ``<fdr_method>fdr_`` before the UUID.

    Parameters
    ----------
    old_filename : str
        Original filename containing a 32-character hex UUID.
    fdr_method : str
        FDR correction method label (e.g. ``"bh"``).

    Returns
    -------
    str
        New filename of the form ``r_<prefix><fdr_method>fdr_<uuid>.npy``.

    Raises
    ------
    ValueError
        If no 32-character hex UUID is found in ``old_filename``.
    """
    # Extract the UUID from the old filename
    uuid_match = re.search(r"([a-f0-9]{32})", old_filename)
    if not uuid_match:
        raise ValueError("UUID not found in the filename")
    uuid = uuid_match.group(1)

    # Extract the rest of the filename before the UUID
    prefix = old_filename.split(uuid)[0]

    # Construct the new filename
    new_filename = f"r_{prefix}{fdr_method}fdr_{uuid}.npy"

    return new_filename


def add_param_to_filename(old_filename, param_name, param_value):
    """
    Insert a hyperparameter name-value pair into a filename before its UUID.

    Skips insertion if the pair is already present. Expects the filename to contain
    a 32-character hex UUID.

    Parameters
    ----------
    old_filename : str
        Original filename containing a 32-character hex UUID.
    param_name : str
        Short label for the hyperparameter (e.g. ``"alpha"``).
    param_value : str or float
        Value of the hyperparameter.

    Returns
    -------
    str
        New filename with ``<param_value><param_name>_`` inserted before the UUID.

    Raises
    ------
    ValueError
        If no 32-character hex UUID is found in ``old_filename``.
    """
    # Convert param_value to string if it's not already
    param_value_str = str(param_value)

    # Check if the parameter name and value are already in the filename
    if f"{param_value_str}{param_name}" in old_filename:
        return old_filename

    # Extract the UUID from the old filename
    uuid_match = re.search(r"([a-f0-9]{32})", old_filename)
    if not uuid_match:
        raise ValueError("UUID not found in the filename")
    uuid = uuid_match.group(1)

    # Extract the rest of the filename before the UUID
    prefix = old_filename.split(uuid)[0]

    # Construct the new filename
    new_filename = f"{prefix}{param_value_str}{param_name}_{uuid}.npy"

    return new_filename


def construct_filename(base_filename, params):
    """
    Build a result filename by appending multiple hyperparameter tags.

    Prepends ``r_`` to ``base_filename`` if not already present, then calls
    :func:`add_param_to_filename` for each entry in ``params``.

    Parameters
    ----------
    base_filename : str
        Base filename containing a 32-character hex UUID.
    params : dict
        Mapping of ``param_name`` → ``param_value`` pairs to embed.

    Returns
    -------
    str
        Filename with all parameter tags inserted before the UUID.
    """
    # Ensure the base filename starts with 'r_'
    if not base_filename.startswith("r_"):
        base_filename = "r_" + base_filename

    # Add each parameter to the filename
    for param_name, param_value in params.items():
        base_filename = add_param_to_filename(base_filename, param_name, param_value)

    return base_filename
