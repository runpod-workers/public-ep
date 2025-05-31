# Tests for FLUX DEV

This directory contains tests for the FLUX DEV endpoint.

## Running Tests

To run the tests, ensure the following environment variables are set:

- `RUNPOD_API_KEY`
- `ENDPOINT_ID`


Ensure all dependencies are installed before running the tests:

```bash
pip install pytest pytest-asyncio aiohttp pillow
```

You can then execute the tests using the following command:

```bash
pytest test_txt2img.py -v
```
