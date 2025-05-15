# __init__.py inside the "mypackage" directory

# Initialize a variable for the package
package_version = "1.0.0"

# Control imports
__all__ = ["pandas_ta", "streaming_indicators"]

from . import pandas_ta
from . import streaming_indicators
