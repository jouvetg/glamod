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
- Runs on Swiss servers
- Login with your **SWITCH edu-ID** (your UNIL account)

---

# RenkuLab: the vocabulary

| Renku | What it is for you |
|---|---|
| **Project** `glamod` | The course project, shared by everyone |
| **Launcher** `glamod` | The **Launch** button: starts a machine with Python, IGM and JupyterLab |
| **Session** | **Your** private machine (CPU, RAM, disk), started from the launcher. Only you see it |

Your files live on the **disk of your session**: they stay there as long as the session is **paused**, not deleted.

---

# Step 1: start your session (now)

1. Go to **renkulab.io** → *Log in* → **SWITCH edu-ID** (UNIL account)
2. Open the course project: **[renkulab.io/p/guillaume.jouvet-1/glamod](https://renkulab.io/p/guillaume.jouvet-1/glamod)** (also on the course page)
3. *Launchers* → `glamod` → **Launch**
4. Wait 1–3 min (longer the first time) → JupyterLab opens in your browser
5. File browser (left): `glamod/` = the course material

**Keep this one session for the whole course**: do not launch a new one each week.

---

# Step 2: the tutorial

Open `glamod/session1/1d-tutorial-sia-model.ipynb` → run the cells from top to bottom (**Shift+Enter**)

At the end of the afternoon: ▾ next to *Open* → **Pause** the session (**not** *Delete*)

---

# Step 3: every following afternoon

1. **renkulab.io** → *Dashboard* → *My sessions* (or the course project) → your paused session → **Resume**
2. Your files are back where you left them
3. Get the latest course material: *File → New → Terminal*, then

```bash
cd ~/work/glamod && git pull
```

4. At the end: **Pause** (not *Delete*)

---

# Step 4: back up your work ⚠️

A paused session is **deleted after some time of inactivity** (a few weeks). From Session 2, back up your project **at the end of each afternoon**:

```bash
cd ~/work && tar czf my-glacier.tar.gz --exclude=outputs --exclude=data my-glacier
```

→ right-click `my-glacier.tar.gz` in the file browser → **Download** → put it in your **UNIL OneDrive**

Lost session? Launch a new one, upload the file (⬆ button) to `~/work`, then `cd ~/work && tar xzf my-glacier.tar.gz`. Run folders (`outputs/`) are re-created by re-running IGM.

---

# If something goes wrong

| Symptom | Fix |
|---|---|
| The session does not start after 5 min | ▾ → *Delete*, launch again; still stuck: call us |
| I cannot find my session | *Dashboard* → *My sessions*; are you logged in with your edu-ID? |
| Kernel dies / very slow | *Kernel → Restart*; close the notebooks you do not use |
| `git pull` refuses (local changes) | `git stash` then `git pull` (or call us) |
