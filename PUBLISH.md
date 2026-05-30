# Publishing clap_family to PyPI

Step-by-step for the maintainer.

## 1. Create and push the GitHub repository

```bash
gh repo create clap-family --public --source=. --remote=origin --push
```

Or manually:

```bash
git remote add origin https://github.com/MrRobotop/clap-family.git
git push -u origin main
```

## 2. Configure PyPI Trusted Publishing (recommended — no token stored)

1. Log in to [pypi.org](https://pypi.org) and register the project name `clap_family`.
2. Go to **Your account → Publishing → Add a new pending publisher**.
3. Fill in:
   - **PyPI project name:** `clap_family`
   - **Owner:** `MrRobotop`
   - **Repository name:** `clap-family`
   - **Workflow name:** `publish.yml`
   - **Environment name:** (leave blank)
4. Save. The `publish.yml` workflow will now authenticate via OIDC on every tag push.

## 3. Tag a release

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `publish.yml` CI job will run `python -m build` and upload to PyPI automatically.

## 4. Fallback: token-based upload

If Trusted Publishing is not configured, use a PyPI API token:

```bash
.venv/bin/python -m build
.venv/bin/twine upload dist/*
```

Enter your PyPI username (`__token__`) and the API token when prompted.

## 5. Verify the install

```bash
python3 -m venv /tmp/verify_clap
/tmp/verify_clap/bin/pip install clap_family
/tmp/verify_clap/bin/python examples/quickstart.py
```

Expected output:

```
target dwell: 0.xxx
trap dwell:   0.000
unsafe time:  0.000
```

## 6. Bump the version for future releases

1. Update `version` in `pyproject.toml` and `__version__` in `clap_family/__init__.py`.
2. Update `version` in `CITATION.cff`.
3. Commit, tag, push.
