PYTHON_VERSION = read_config("blckbaux", "python_version", "3.13")

_PYTHON_CONFIGS = {
    "3.12": struct(
        abi = "cp312",
        environment = "blckbaux-py312",
        extension_infix = "312",
    ),
    "3.13": struct(
        abi = "cp313",
        environment = "blckbaux",
        extension_infix = "313",
    ),
    "3.14": struct(
        abi = "cp314",
        environment = "blckbaux-py314",
        extension_infix = "314",
    ),
    "3.15": struct(
        abi = "cp315",
        environment = "blckbaux-py315",
        extension_infix = "315",
    ),
}
if PYTHON_VERSION not in _PYTHON_CONFIGS:
    fail("blckbaux.python_version must be 3.12 through 3.15, got {}".format(PYTHON_VERSION))

_PYTHON = _PYTHON_CONFIGS[PYTHON_VERSION]

PYTHON_ABI = _PYTHON.abi
PYTHON_ENVIRONMENT = _PYTHON.environment
PYTHON_REQUIRES = ">=3.12,<3.16"

def python_extension_suffix(platform):
    if platform == "linux-aarch64":
        return ".cpython-{}-aarch64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "linux-x86_64":
        return ".cpython-{}-x86_64-linux-gnu.so".format(_PYTHON.extension_infix)
    if platform == "macos-arm64":
        return ".cpython-{}-darwin.so".format(_PYTHON.extension_infix)
    fail("unsupported Python extension platform: {}".format(platform))
