---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Session 1 — Hour 3: Setting up RenkuLab, then the tutorial

---

# Where do we compute?

On [renkulab.io](https://renkulab.io), a platform of the Swiss Data Science Center (EPFL/ETH)

- JupyterLab **in your browser**: nothing to install on your laptop
- Python, IGM and all packages are **pre-installed** in the course environment
- Runs on Swiss servers, with resources reserved for this course
- Login with your **SWITCH edu-ID** (UNIL account)

---

# RenkuLab: the vocabulary

| Renku | What it is for you |
|---|---|
| **Project** | Your copy of the course: material (code repository) + launcher + storage |
| **Session launcher** | The "Launch" button: starts a machine with Python, IGM and JupyterLab |
| **Session** | Your running machine (CPU, RAM, disk): temporary! |
| **Data connector** | Your SWITCHdrive folder, mounted in the session: **where you save your work** |

---

# Step 1: your own copy of the course (once, now)

1. Go to **renkulab.io** → *Log in* with SWITCH edu-ID
2. Open the course project (link on Moodle)
3. Click the banner button to **copy the project** → this is now *your* project
4. In your project: add a **data connector** → *SWITCHdrive* → choose a folder (e.g. `glamod-work`)
5. *Settings* → *Members*: add the teacher and the assistant as **viewers**

---

# Step 2: launch a session (every afternoon)

1. Open **your** project → *Sessions* → **Launch** (`glamod` launcher)
2. Wait 1–3 min (longer the first time) → JupyterLab opens in your browser
3. File browser (left): `glamod/` = course material, plus your SWITCHdrive folder
4. Open `glamod/session1/1d-tutorial-sia-model.ipynb` → run the cells top to bottom (Shift+Enter)

At the end of the afternoon: **Pause** the session (do not delete it).

---

# Step 3: save your work ⚠️

A session is a **temporary machine**: it may be deleted after a period of inactivity.

- Copy what matters to your **SWITCHdrive folder** (drag & drop in the file browser), or right-click → *Download*
- **Rule**: never finish an afternoon without saving your notebooks (and later your YAML files and figures)

| Symptom | Fix |
|---|---|
| Session does not start | Wait, retry (or choose the CPU resource class); check you launched from **your** project |
| Kernel dies / very slow | Restart the kernel (*Kernel → Restart*); close other notebooks |
| Lost files | Were they in the SWITCHdrive folder? |
