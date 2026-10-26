# Working online with RenkuLab

All computations of this course run **online** on [RenkuLab](https://renkulab.io), the Swiss Data Science Center platform (ETH/EPFL). You only need a web browser: IGM and all Python packages are pre-installed. Nothing to install on your laptop.

## For students

The same guide is presented in the slides `session1/1c-setup-renku.md` (Session 1, hour 3: setup, then Tutorial 1). How to work with IGM is presented in Lecture 2.

| Renku | What it is for you |
|---|---|
| **Project** | Your copy of the course: material (code repository) + launcher + storage |
| **Session launcher** | The "Launch" button: starts a machine with IGM + JupyterLab |
| **Session** | Your running machine (CPU, RAM, disk) — temporary! |
| **Data connector** | Your SWITCHdrive folder, mounted in the session: where you **save** your work |

### Step 1 — Your own copy of the course (once, Session 1, hour 3)

1. Go to [renkulab.io](https://renkulab.io) and log in with your **SWITCH edu-ID** (UNIL account).
2. Open the course project (**link on Moodle**).
3. Click the button in the banner to **copy the project**: you now have your own project.
4. In your project, add a **data connector** to a **SWITCHdrive** folder (e.g. create `glamod-work` in SWITCHdrive first). This is where you save your work.
5. *Settings* → *Members*: add the teacher and the assistant (viewer access), so that we can help you.

### Step 2 — Launch a session (every afternoon)

1. Open **your** project → *Sessions* → **Launch** on the `glamod` launcher. Starting takes 1–3 minutes.
2. JupyterLab opens in your browser. The file browser shows the course material (`glamod/`) and your SWITCHdrive folder:

```
glamod/
├── session1/     lectures, 1d-tutorial-1-sia-model.ipynb, 1e-choose-your-glacier.md
├── session2/     lecture on IGM, tutorial-2/ (IGM hands-on)
├── session3/     project goals & report guidelines
├── session4/     Mont Miné (course "Dates and Rates")
├── project/      mini-project guide + my-glacier/ (your project template)
└── utils/        plotting helpers used by the notebooks
```

3. At the end of the afternoon: **Pause** the session (do not delete it).

### Step 3 — Work with IGM: YAML + terminal + notebook (Sessions 2–3)

| What | Where |
|---|---|
| **Set** the parameters | YAML files in `experiment/`, opened in the JupyterLab text editor |
| **Run** IGM | a JupyterLab **Terminal** (*File → New → Terminal*) |
| **Look** at the results | a **notebook** that only reads the run folders (`analysis.ipynb` in the project) |

Drag the tabs to put them side by side. In the terminal, from the folder that contains `experiment/`:

```bash
cd ~/work/glamod/session2/tutorial-2
igm_run +experiment=params_A_aletsch hydra.run.dir=outputs/A_reference
```

Results are written to the named folder `outputs/A_reference/` (`output.nc`, `output_ts.nc`, `dashboard_XXXX.png`, `.hydra/` with all parameters). A run in the terminal keeps going if you close the browser tab. Stop it with Ctrl+C.

### Step 4 — Save your work ⚠️

**A session is a temporary machine.** Paused sessions keep their files, but sessions that stay inactive for too long are deleted.

- Copy your notebook(s), your `experiment/*.yaml` files and your figures to your **SWITCHdrive folder** (drag & drop in the file browser, or `cp -r` in a terminal), or right-click → *Download*.
- `output.nc` files can be large: keep only the useful ones. They can be re-created by re-running IGM.
- **Never finish an afternoon without saving your YAML files and your notebook.**

### If something goes wrong

| Symptom | Fix |
|---|---|
| The session does not start | Wait, retry. Check that you launched from **your** project. |
| `igm_run: command not found` | Wrong launcher: use the `glamod` launcher. |
| `Parameter 'experiment' does not exist` | You are not in the folder that contains `experiment/`: `cd` there. |
| The kernel dies / very slow | Out of memory: restart the kernel, shorten the run, close other notebooks. |
| Data download (`oggm_shop`) fails | Retry. The data comes from a server in Bremen, which may be slow when 20 students download at once. |
| Lost files | Were they saved in the SWITCHdrive folder? |

---

## For the teaching team (setup before the course)

### Strategy ("classic mode" of Renku for teaching)

- **Content**: the GitHub repository `jouvetg/glamod`, added to the Renku project as a *code repository* (cloned into each session).
- **Environment**: the Docker image `ghcr.io/jouvetg/glamod:2026` (`environment/Dockerfile`: Jupyter + IGM from its main branch, pinned commit in `environment/requirements.txt`), built by GitHub Actions (`.github/workflows/build-image.yml`).
- **Project**: one Renku 2.0 project, marked as a **template**. Each student copies it (with the code repository and the launcher) and adds their own SWITCHdrive data connector to save their work.
- Docs: [Renku for teaching](https://docs.renkulab.io/en/2.14.1/docs/users/use-cases/teaching/), [set up a course in classic mode](https://docs.renkulab.io/en/2.14.1/docs/users/use-cases/teaching/set-up-course-in-classic-mode), [use your own Docker image](https://docs.renkulab.io/en/2.18.1/docs/users/sessions/guides/environments/use-your-own-docker-image-for-renku-session).

### Steps

1. **Request a course resource pool** from the Renku team (Contact page on renkulab.io), well in advance: course name, dates of the 3 afternoons (+ project work in between), number of students (max. 15) + teaching team (2), and the list of participants. Ask for **GPU sessions**: IGM runs much faster on a GPU (the image supports both, nothing to change). Points to discuss with them:
   - **how many GPUs**: 15 students + 2 teaching team, **3 sessions per GPU** → **5–6 GPUs** shared between sessions (time-slicing or MIG slices). Sharing is fine for our small glaciers: the image sets `TF_FORCE_GPU_ALLOW_GROWTH=true`, so a session only takes the GPU memory it needs (to be measured during the dry run with `nvidia-smi`);
   - per session: 1 GPU (or slice), 4 CPUs, 16 GB RAM, 20 GB disk (the image is ~9 GB);
   - NVIDIA driver on the nodes: compatible with CUDA 12.3 (TensorFlow 2.17);
   - what happens when all GPUs are busy (sessions wait in a queue) → keep a **CPU resource class** in the same pool as a fallback (4 CPUs, 8–16 GB RAM): everything also runs on CPU, only slower (see timings below).
2. **Publish the image**: push the repo, check that the GitHub Action ran, and make the `glamod` package **public** on GitHub (Packages → glamod → Package settings → Change visibility). RenkuLab can only pull public images.
3. **(Optional) Create a group** for the course on renkulab.io: teaching team = owners, students = viewers.
4. **Create the project** (in the group) → add the **code repository** `https://github.com/jouvetg/glamod` (public).
5. **Add a session launcher**: *Sessions* → ➕ → **External environment**:
   - **Container image**: `ghcr.io/jouvetg/glamod:2026`
   - **Advanced settings**:
     - Default URL: `/lab`
     - Port: `8888`
     - Mount directory: `/home/jovyan/work`
     - Working directory: `/home/jovyan/work`
     - UID: `1000`, GID: `100` (the `jovyan` user of the Jupyter images)
     - Command ENTRYPOINT: `["sh", "-c"]`
     - Command arguments (as given by the Renku docs for Jupyter images):
       ```
       ["jupyter server --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --ServerApp.base_url=$RENKU_BASE_URL_PATH --ServerApp.token=\"\" --ServerApp.password=\"\" --ServerApp.allow_remote_access=true --ContentsManager.allow_hidden=true --ServerApp.allow_origin=*"]
       ```
   - **Resource class**: the GPU class of the course resource pool (students can pick the CPU class at launch if no GPU is free).
   - Name: `glamod` → *Add session launcher*.
6. **Mark the project as a template**: *Settings* → toggle *Template project*. Make it visible to the students (public, or within the course group), and post the link on Moodle.
7. **Dry run** with the assistant (as a student account): copy the template, add a SWITCHdrive data connector, launch, run `session1/1d-tutorial-1-sia-model.ipynb`, `session2/tutorial-2/tutorial-2-igm.ipynb` and the project (`project/my-glacier/`: check run, `params_past`, `bash run_future.sh`, then `analysis.ipynb`), save a file to SWITCHdrive, pause and resume the session. Check that the GPU is used (`nvidia-smi` in a terminal while IGM runs), measure the times against the CPU table below, and check the idle/hibernation rules of the resource pool with the Renku team.

**Updating the material during the course**: students' copies keep a link to the GitHub repository; a new session clones the latest version. In an existing (paused) session, run `git pull` in `~/work/glamod` (students' own changes may conflict: they should work in copies of the files, e.g. in their SWITCHdrive folder or with new file names).

### Timings measured on CPU (IGM 3.2.0 + emulators, laptop CPU, no GPU) — GPU timings to be measured during the dry run

| Run | What | Wall time |
|---|---|---|
| `params_A_aletsch` | Aletsch, 100 m, 1900–2000, simple SMB, off-line emulator | ~80 s |
| `params_B_any_glacier` | data download + Findelen, 2020–2100, OGGM climate SMB | ~80 s |
| `params_C_custom_smb` | Aletsch, 1900–2300, user SMB module | ~4 min |
| project `params_past` | Findelen, 1700–2020, climate SMB | ~4.5 min |
| project `params_past_simple` | Findelen, 1700–2020, ELA SMB | ~4.5 min |
| project `params_future` | Findelen, 2020–2100, climate SMB | ~1 min |

Larger glaciers take somewhat longer (for reference, Aletsch, 82 km², runs 100 years in ~80 s). These timings used ~8 CPU cores (TensorFlow uses all available cores): with 2 CPUs per session, expect runs 2–4× slower. Re-measure during the dry run, and size the resource class accordingly (4 CPUs per session would be comfortable). Executing the whole of Tutorial 2, or the whole project notebook (past + 4 scenarios), took ~9 min each on 8 cores. 20 students running at the same time need a correspondingly sized resource pool.

### Check the glacier list

Before the course, check that all candidate RGI IDs download and run (from `project/my-glacier/`):

```bash
for id in $(python -c "import pandas as pd; print(' '.join(pd.read_csv('../../session1/candidate-glaciers.csv').rgi_id))"); do
  igm_run +experiment=params_future inputs.oggm_shop.RGI_ID=$id processes.time.end=2021 \
     hydra.run.dir=outputs/check_$id > check_$id.log 2>&1 && echo "OK $id" || echo "FAIL $id"
done
```

### Fallback options

- **Local install** (students with a Linux/Mac laptop, or WSL on Windows): `pip install igm-model==3.2.1` in a Python 3.11 environment. See [igm-model.org](https://igm-model.org/latest/installation/quick_start/).
- **Docker** on any machine: `docker run -p 8888:8888 ghcr.io/jouvetg/glamod:2026`.
- **UNIL JupyterHub / Curnagl cluster**: the same image or requirements can be used.
