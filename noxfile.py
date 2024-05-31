import os
import pathlib
import shutil
from typing import Iterable

import nox

nox.options.envdir = ".cache/nox"


def _get_conda_binary_name() -> str:
    """
    Checks if conda and / or mamba is installed and returns binary name.

    We prefer mamba for speed of installation where available.
    """
    mamba_path = shutil.which("mamba")
    if mamba_path:
        print(f"Found mamba installation '{mamba_path}'")
        return "mamba"
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"Found conda installation '{conda_path}'")
        return "conda"
    raise RuntimeError(
        "This repo is configured to use Conda for nox sessions, but could not find"
        " conda or mamba in $PATH.\n"
    )


CONDA_OR_MAMBA = _get_conda_binary_name()
nox.options.default_venv_backend = CONDA_OR_MAMBA
nox.options.reuse_existing_virtualenvs = False
# Define what runs by default if you just type nox;
# these should match the PR checks in bitbucket-pipelines.yml.
nox.options.sessions = [
    "formatting_check",
    "linting_check",
    "typing_check",
    "dependencies_check",
    "tests_run_pinned",
]


# List some source paths. See also settings for black,mypy in pyproject.toml.
LINT_SOURCE_PATHS = (
    "src/",
    "tests/",
    "noxfile.py",
)
# Sphinx-config has non-standard conventions, doesn't play nicely with pylint,
# so we add it just for isort.
# (Keep this in sync with `tool.black` section of pyproject.toml)
ISORT_SOURCE_PATHS = LINT_SOURCE_PATHS + ("docs/source/conf.py",)
# Autoformat tools: These will run fine without the base package installed, so we list them here
# (to keep it DRY / avoid needing to parse setup.cfg).
FORMATTING_TOOLS = ("black==22.3.0", "isort")  # keep in sync with pre-commit-config!

# Dependency pinning:
CONSTRAINTS_PINNED_PATH = pathlib.Path("pinned-versions.txt")
CONSTRAINTS_LATEST_PATH = pathlib.Path("latest-versions.txt")  # should be .gitignored
# whitelist / ignore the following when checking pinned versions
# (This allows us to compare results from pip-compile across e.g. linux / macosx builds)
PLATFORM_DEPENDENT_PACKAGES_WHITELIST = (
    "appnope",  # package for disabling App Nap on macOS (installed by iPython)
    "importlib-resources",  # backports some utils for python<3.9.
    "importlib-metadata",  # backports some utils for python<3.10.
    "zipp",  # dependency of the importlib- entries above.
    # "openeye-toolkits-python3-linux-x64",  # uncomment these if openeye is a dependency.
    # "openeye-toolkits-python3-osx-x64",
)
# Default set of extras to select when pinning dependencies.
PACKAGE_EXTRAS_TO_PIN: Iterable[str] = ("docs", "tests", "typing")
PYTHON_DEFAULT = "3.9"

PROJECT_DIR = pathlib.Path(__file__).parent


def _conda_prep_env(session: nox.Session) -> None:
    session.run_always(
        *[
            CONDA_OR_MAMBA,
            "env",
            "update",
            "--prefix",
            session.virtualenv.location,
            "--file",
            "environment_cpu.yml",
        ],
        silent=True,
    )


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def dist_build(session: nox.Session) -> None:
    """Build distributions (sdist and wheel).

    The distribution packages are built using PyPA's `build`_ build frontend.
    This is the recommended way of building python packages, avoiding direct
    calls to the build backend. Legacy calls like ``$ python setup.py build``
    are now deprecated.

    .. _build:
            https://pypa-build.readthedocs.io/en/latest/
    """

    session.run("rm", "-rf", "dist", external=True)
    session.install("build")
    session.run("python", "-m", "build")


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def formatting_check(session: nox.Session) -> None:
    """
    Checks your code is formatted correctly (does not make changes).
    """
    session.install(*FORMATTING_TOOLS)
    session.run("black", "--check", "--diff", ".")
    session.run("isort", "--check", "--diff", *ISORT_SOURCE_PATHS)


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def formatting_fix(session: nox.Session) -> None:
    """
    Runs black and isort to autoformat your code.
    """
    session.install(*FORMATTING_TOOLS)
    session.run("black", ".")
    session.run("isort", *ISORT_SOURCE_PATHS)


@nox.session(python=PYTHON_DEFAULT)
def typing_check(session: nox.Session) -> None:
    """
    Checks your code using the mypy type-checker.
    """
    # NB mypy requires full deps install to get full type hints.
    _conda_prep_env(session)
    session.install(".[tests,typing]", "--constraint", str(CONSTRAINTS_PINNED_PATH))
    session.run("mypy")


@nox.session(python=PYTHON_DEFAULT)
def linting_check(session: nox.Session) -> None:
    """
    Checks your code using linting tools (pylint).
    """
    _conda_prep_env(session)
    session.install(".[linting,tests]", "--constraint", str(CONSTRAINTS_PINNED_PATH))
    session.run("pylint", *LINT_SOURCE_PATHS)


def _tests_run(session: nox.Session, constraints_file: pathlib.Path) -> None:
    _conda_prep_env(session)
    session.install(".[tests]", "--constraint", str(constraints_file))
    output_file = ".coverage"
    session.run("rm", "-rf", f"{output_file}", external=True)

    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "--ignore",
        "tests/openfold/",
        "--ignore",
        "tests/loss/",
        *session.posargs,
    )
    session.notify("coverage_report")


@nox.session(python=PYTHON_DEFAULT)
def tests_run_latest(session: nox.Session) -> None:
    """
    Test your code against the latest dependencies.

    (Will install latest versions compatible with the specifications in `setup.cfg`.)
    """
    _dependencies_pin(
        session=session,
        constraints_path=CONSTRAINTS_LATEST_PATH,
        upgrade=True,
        use_posargs=False,
    )
    _tests_run(session, constraints_file=CONSTRAINTS_LATEST_PATH)


@nox.session(python=PYTHON_DEFAULT)
def tests_run_pinned(session: nox.Session) -> None:
    """
    Test your code against stable dependencies.

    (Will install dependencies as specified in `pinned-versions.txt`.)
    """
    _tests_run(session, constraints_file=CONSTRAINTS_PINNED_PATH)


def _dependencies_pin(
    session: nox.Session,
    constraints_path: pathlib.Path,
    upgrade: bool,
    use_posargs: bool,
    extras: Iterable[str] = PACKAGE_EXTRAS_TO_PIN,
) -> None:
    session.install("pip-tools")
    extras_str = ",".join(extras)
    session_args = [
        "pip-compile",
        f"--extra={extras_str}",
        "--strip-extras",  # compatibility with pip >= 20.3
        "--no-emit-index-url",  # avoids committing your PyPI token!
        "--no-emit-trusted-host",  # avoid diff churn
        "setup.cfg",
        "-o",
        str(constraints_path),
    ]
    if upgrade:
        session_args.append("--upgrade")
    if use_posargs:
        session_args.extend(session.posargs)
    session.run(*session_args)


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def dependencies_pin(session: nox.Session) -> None:
    """
    Recompile pinned-versions.txt, making minimal changes.
    """
    _dependencies_pin(
        session=session,
        constraints_path=CONSTRAINTS_PINNED_PATH,
        upgrade=False,
        use_posargs=True,
    )


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def dependencies_pin_upgrade_all(session: nox.Session) -> None:
    """
    Recompile pinned-versions.txt, upgrading packages wherever possible.
    """
    _dependencies_pin(
        session=session,
        constraints_path=CONSTRAINTS_PINNED_PATH,
        upgrade=True,
        use_posargs=True,
    )


@nox.session(python=PYTHON_DEFAULT, reuse_venv=True)
def dependencies_check(session: nox.Session) -> None:
    """
    Check pinned-versions.txt is in-sync with setup.cfg.

    (Generates throwaway file 'pinned-versions.check' and compares with pinned-versions.txt.)
    """
    # local import because pkg_resources is notoriously slow to load, see e.g.
    # https://github.com/pypa/setuptools/issues/926
    import pkg_resources  # pylint: disable=import-outside-toplevel

    # generate a new, temporary constraints file
    temp_constraints_path = CONSTRAINTS_PINNED_PATH.with_suffix(".check")
    temp_constraints_path.unlink(missing_ok=True)  # delete any old copy
    shutil.copy(
        src=CONSTRAINTS_PINNED_PATH,
        dst=temp_constraints_path,
    )
    _dependencies_pin(
        session=session,
        constraints_path=temp_constraints_path,
        upgrade=False,
        use_posargs=True,
    )
    # now compare old and new
    with CONSTRAINTS_PINNED_PATH.open() as f:
        old_requirements = set(pkg_resources.parse_requirements(f))
    with temp_constraints_path.open() as f:
        new_requirements = set(pkg_resources.parse_requirements(f))
    mismatch_packages = new_requirements.symmetric_difference(old_requirements)
    filtered_mismatches = set(
        pkg
        for pkg in mismatch_packages
        if pkg.key not in PLATFORM_DEPENDENT_PACKAGES_WHITELIST
    )
    if filtered_mismatches:
        session.error(
            f"Got the following {len(filtered_mismatches)} version mismatches:\n "
            f"{sorted(filtered_mismatches, key=lambda x: x.key)}"
        )


@nox.session(python=PYTHON_DEFAULT)
def notebooks_convert_to_percent(session: nox.Session) -> None:
    """Converts all notebooks in given folder to percent format.

    Defaults to converting all notebooks found in the ``notebooks/`` directory.
    """

    if session.posargs:
        notebooks_dir = session.posargs[0]
    else:
        notebooks_dir = "notebooks"

    session.install(".[docs]", "--constraint", str(CONSTRAINTS_PINNED_PATH))

    cmd = f"jupytext --from ipynb --to py:percent {notebooks_dir}/*.ipynb"
    session.run(*cmd.split(" "))

    for jupytext in pathlib.Path(notebooks_dir).glob("*.py"):
        jupytext.replace(jupytext.with_suffix(".pct.py"))
