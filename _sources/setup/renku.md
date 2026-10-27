# Working online with RenkuLab

All computations run **online** on [RenkuLab](https://renkulab.io) (Swiss Data Science Center, ETH/EPFL), in a web browser: IGM and all Python packages are pre-installed, nothing to install on your laptop. Everything runs on CPU.

## For students

Presented in the slides `session1/1c-setup-renku.md` (Session 1, hour 3).

| Renku | What it is for you |
|---|---|
| **Project** | Your copy of the course: material + launcher + storage |
| **Session** | Your running machine with IGM + JupyterLab: **temporary!** |
| **Data connector** | Your SWITCHdrive folder, where you **save** your work |

### Once (Session 1)

1. Log in to [renkulab.io](https://renkulab.io) with your **SWITCH edu-ID**.
2. Open the course project (**link on Moodle**) and **copy** it (button in the banner).
3. In your copy, add a **data connector** to a **SWITCHdrive** folder (e.g. `glamod-work`).
4. *Settings* → *Members*: add the teacher and the assistant (viewer).

### Every afternoon

1. **Your** project → *Sessions* → **Launch** (`glamod` launcher, 1–3 min). JupyterLab opens with the course material in `~/work/glamod/` and your SWITCHdrive folder. Your project work lives in your own copy `~/work/my-glacier/` (made once in Session 2, Lecture 2).
2. Work with **YAML** files (parameters) + a **terminal** (`igm_run ...`) + a **notebook** (plots): see Lecture 2.
3. **Save** your YAML files, notebooks, scripts and figures to SWITCHdrive (drag & drop, or `cp -r`). Run folders can be re-created by re-running IGM.
4. At the end: **Pause** the session (do not delete it). Sessions inactive for too long are deleted: never leave without saving.

### If something goes wrong

| Symptom | Fix |
|---|---|
| The session does not start | Wait, retry; check that you launched from **your** project |
| `igm_run: command not found` | Use the `glamod` launcher |
| `Parameter 'experiment' does not exist` | `cd` to the folder that contains `experiment/` |
| Kernel dies / very slow | Out of memory: restart the kernel, close other notebooks |
| Data download (`oggm_shop`) fails | Retry (the server may be busy when everyone downloads at once) |

---

## For the teaching team

**Set-up**: the GitHub repository `jouvetg/glamod` is the *code repository* of one Renku 2.0 **template project**; the session runs the Docker image `ghcr.io/jouvetg/glamod:2026` (`environment/`, IGM pinned to a commit of its main branch), built by GitHub Actions on every change in `environment/`. Docs: [Renku for teaching](https://docs.renkulab.io/en/2.14.1/docs/users/use-cases/teaching/), [own Docker image](https://docs.renkulab.io/en/2.18.1/docs/users/sessions/guides/environments/use-your-own-docker-image-for-renku-session).

### Steps

1. **Image**: push the repo, check the GitHub Action, make the `glamod` package **public** (GitHub → Packages → glamod → Package settings).
2. **Project** (optionally in a course group): add the code repository `https://github.com/jouvetg/glamod`.
3. **Session launcher**: *Sessions* → ➕ → **External environment**, image `ghcr.io/jouvetg/glamod:2026`, name `glamod`, CPU resource class (4 CPUs, 8–16 GB RAM). Advanced settings:
   - Default URL `/lab`, port `8888`, mount and working directory `/home/jovyan/work`, UID `1000`, GID `100`
   - Command ENTRYPOINT `["sh", "-c"]`, arguments:
     ```
     ["jupyter server --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --ServerApp.base_url=$RENKU_BASE_URL_PATH --ServerApp.token=\"\" --ServerApp.password=\"\" --ServerApp.allow_remote_access=true --ContentsManager.allow_hidden=true --ServerApp.allow_origin=*"]
     ```
4. **Template**: *Settings* → *Template project*, make it visible to the students, post the link on Moodle.
5. **Dry run** with a student account: copy, connect SWITCHdrive, launch, run the tutorial, copy `session2/my-glacier` to `~/work/my-glacier` and run the project steps (check run, `params_past`, `bash run_future.sh`, `analysis.ipynb`), save to SWITCHdrive, pause and resume.

**Updating during the course**: the **image** only holds the software (IGM + packages): build it once, and rebuild only if `environment/` changes (new IGM commit). The **course material** comes from the GitHub repository: push changes any time; a new session clones the latest version, and in a paused session students run `git pull` in `~/work/glamod`. Their own work in `~/work/my-glacier/` is never touched by a pull; changes to the template `session2/my-glacier/` after they copied it must be copied by hand (e.g. a new script: `cp ~/work/glamod/session2/my-glacier/run_xxx.sh ~/work/my-glacier/`).

### Timings (CPU, 8 cores, `dahunet_edu`, coarsening 2)

| Run | Wall time |
|---|---|
| Project (Findelen): check / past 1700–2020 / `run_future.sh` (4 scenarios) | ~12 s / ~50 s / ~1.5 min |
| Project: response time / moraines (particles) / `run_sensitivity.sh` | ~1 min / ~2 min / ~2.5 min |

With fewer cores, expect runs 2–4× slower; with `coarsening: ratio: 1`, ~1.5–2× slower.

### Check the glacier list

From `session2/my-glacier/`, check that all candidate RGI IDs download and run:

```bash
for id in $(python -c "import pandas as pd; print(' '.join(pd.read_csv('../../session1/candidate-glaciers.csv').rgi_id))"); do
  igm_run +experiment=params_future inputs.oggm_shop.RGI_ID=$id processes.time.end=2021 \
     hydra.run.dir=outputs/check_$id > check_$id.log 2>&1 && echo "OK $id" || echo "FAIL $id"
done
```

### Fallbacks

- **Docker** on any machine: `docker run -p 8888:8888 ghcr.io/jouvetg/glamod:2026`
- **Local install** (Linux/Mac/WSL, Python 3.11): `pip install -r environment/requirements.txt`
