import re


def add_fdr_to_filename(old_filename, fdr_method):
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
    Add a hyperparameter name and value to the filename before the UUID.

    Parameters:
    old_filename (str): The original filename.
    param_name (str): The name of the hyperparameter.
    param_value (str or float): The value of the hyperparameter.

    Returns:
    str: The new filename with the hyperparameter name and value.
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
    Construct a filename with the given parameters.

    Parameters:
    base_filename (str): The base filename.
    params (dict): A dictionary of parameter names and values.

    Returns:
    str: The constructed filename with the parameters.
    """
    # Ensure the base filename starts with 'r_'
    if not base_filename.startswith("r_"):
        base_filename = "r_" + base_filename

    # Add each parameter to the filename
    for param_name, param_value in params.items():
        base_filename = add_param_to_filename(base_filename, param_name, param_value)

    return base_filename
