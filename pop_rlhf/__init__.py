# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preference over Preferences (PoP): adaptive-margin DPO. Built on the Safe-RLHF framework."""

# Kept lightweight on purpose: sub-packages (algorithms, trainers, ...) import DeepSpeed, so import them
# explicitly (e.g. `from pop_rlhf.datasets import ...`) instead of at package import time.
from pop_rlhf.version import __version__

__all__ = ['__version__']
