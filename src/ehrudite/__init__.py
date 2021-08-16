import sys

# Expose the public API.
# Nothing to add

# Check major python version
if sys.version_info[0] < 3:
    raise Exception("Ehrudite does not support Python 2. Please upgrade to Python 3.")
# Check minor python version
elif sys.version_info[1] < 6:
    raise Exception(
        "Ehrudite only supports Python 3.6 and beyond. " "Use a later version of Python"
    )

# Set the version attribute of the library
import pkg_resources
import configparser

# Get the current version
config = configparser.ConfigParser()
config.read([pkg_resources.resource_filename("ehrudite", "config.ini")])

__version__ = config.get("ehrudite", "version")
