import functools
import logging
import os
import sys
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic.utils import deep_update

logger = logging.getLogger(__name__)

PROJECT_ROOT_PATH: Path = Path(__file__).parents[2]  # Root of the project

# The location of the settings folder. Defaults to the project root
_settings_folder = os.environ.get("RAG_BENCH_SETTINGS_FOLDER", PROJECT_ROOT_PATH)

# If running in unittest, use the test profile
_test_profile = ["test"] if "pytest" in sys.modules else []

active_profiles: list[str] = list(
    set(
        ["default"]
        + [
            item.strip()
            for item in os.environ.get("RAG_BENCH_PROFILES", "").split(",")
            if item.strip()
        ]
        + _test_profile
    )
)


def load_yaml_with_envvars(stream: TextIO) -> dict[str, Any]:
    """Load yaml file with environment variable expansion.

    The pattern ${VAR} or ${VAR:default} will be replaced with
    the value of the environment variable.
    """
    def env_var_replacer(value):
        """Replace environment variables in string values."""
        if not isinstance(value, str):
            return value
            
        pattern = r'\$\{([^}^{]+)\}'
        matches = re.findall(pattern, value)
        
        if not matches:
            return value
            
        for match in matches:
            env_var = match
            default = None
            
            if ':' in match:
                env_var, default = match.split(':', 1)
                
            env_value = os.environ.get(env_var)
            
            if env_value is None and default is None:
                raise ValueError(f"Environment variable {env_var} is not set and no default was provided")
                
            replace_value = env_value if env_value is not None else default
            value = value.replace(f"${{{match}}}", replace_value)
            
        return value

    def process_mapping(mapping):
        """Process all string values in a mapping to replace env vars."""
        result = {}
        for k, v in mapping.items():
            if isinstance(v, dict):
                result[k] = process_mapping(v)
            elif isinstance(v, list):
                result[k] = [
                    process_mapping(item) if isinstance(item, dict) 
                    else env_var_replacer(item) 
                    for item in v
                ]
            else:
                result[k] = env_var_replacer(v)
        return result

    # Load the YAML normally
    raw_config = yaml.safe_load(stream)
    
    # Process environment variables in the loaded config
    if isinstance(raw_config, dict):
        return process_mapping(raw_config)
    return raw_config


def merge_settings(settings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple settings dictionaries, with later dictionaries taking precedence."""
    return functools.reduce(deep_update, settings, {})


def load_settings_from_profile(profile: str) -> dict[str, Any]:
    """Load settings from a specific profile."""
    if profile == "default":
        profile_file_name = "settings.yaml"
    else:
        profile_file_name = f"settings-{profile}.yaml"

    path = Path(_settings_folder) / profile_file_name
    try:
        with Path(path).open("r") as f:
            config = load_yaml_with_envvars(f)
        if not isinstance(config, dict):
            raise TypeError(f"Config file has no top-level mapping: {path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Settings file not found: {path}")
        return {}


def load_active_settings() -> dict[str, Any]:
    """Load active profiles and merge them."""
    logger.info("Starting application with profiles=%s", active_profiles)
    loaded_profiles = [
        load_settings_from_profile(profile) for profile in active_profiles
    ]
    merged: dict[str, Any] = merge_settings(loaded_profiles)
    return merged


def load_settings(path: str) -> dict[str, Any]:
    """Load settings from a specific path."""
    try:
        with Path(path).open("r") as f:
            config = load_yaml_with_envvars(f)
        if not isinstance(config, dict):
            raise TypeError(f"Config file has no top-level mapping: {path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Settings file not found: {path}. Using default settings.")
        return load_active_settings()
    except Exception as e:
        logger.error(f"Error loading settings from {path}: {e}")
        return load_active_settings()