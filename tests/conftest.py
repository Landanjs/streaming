# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import pytest

from tests.common.utils import compressed_remote_local  # pyright: ignore
from tests.common.utils import get_free_tcp_port  # pyright: ignore
from tests.common.utils import remote_local  # pyright: ignore
from tests.test_reader import mds_dataset_dir  # pyright: ignore


@pytest.fixture(scope='session', autouse=True)
def tests_setup_and_teardown():
    # Will be executed before the first test
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(get_free_tcp_port())


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)  # pyright: ignore
def pytest_runtest_call(item: Any):
    # Launch a custom function for distributed tests
    if getattr(item.cls, 'is_dist_test', False):
        dist_test_class = item.cls()
        dist_test_class._run_test(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


@pytest.fixture(scope='session', autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'


@pytest.fixture(scope='session', autouse=True)
def gcs_credentials():
    """Mocked GCS Credentials for moto."""
    os.environ['GCS_KEY'] = 'testing'
    os.environ['GCS_SECRET'] = 'testing'
