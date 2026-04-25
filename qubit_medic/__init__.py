"""Qubit-Medic: An LLM-trained quantum error-correction decoder.

The package is split into three layers (Section 0 of the plan):

* ``qubit_medic.config``   - the locked experiment configuration.
* ``qubit_medic.server``   - Stim physics, rewards, curriculum, FastAPI app.
* ``qubit_medic.client``   - the lightweight HTTP stub the trainer imports.

``qubit_medic.models`` and ``qubit_medic.prompts`` are the contract both sides
agree on: what the LLM sees and what the LLM is allowed to emit.
"""

from qubit_medic import config, models, prompts  # noqa: F401

__version__ = "1.0.0"
