"""Setup file for the package."""
import os
import re

from setuptools import setup


def read(fname: str) -> str:
  return open(os.path.join(os.path.dirname(__file__), fname),
              encoding="utf-8").read()


def find_version() -> str:
  version_file = read("torchlaplace/version.py")
  version_re = r"__version__ = \"(?P<version>.+)\""
  version_raw = re.match(version_re, version_file)

  if version_raw is None:
    return "0.0.1"

  version = version_raw.group("version")
  return version


if __name__ == "__main__":
  try:
    setup(version=find_version(),)
  except:  # noqa
    print("\n\nAn error occurred while building the project, "
          "please ensure you have the most updated version of setuptools, "
          "setuptools_scm and wheel with:\n"
          "   pip install -U setuptools setuptools_scm wheel\n\n")
    raise
