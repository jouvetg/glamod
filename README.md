# glamod — Glacier Modelling course (UNIL, Master)

Course material for *Glacier Modelling* (G. Jouvet, UNIL): 3 afternoons × 4 hours, each one hour of lecture, about one hour of guided hands-on, then a mini-project on a Swiss glacier with [IGM 3.2](https://igm-model.org). Students run everything online on [RenkuLab](https://renkulab.io).

```
glamod/
├── infos.md                  course front page (programme, links)
├── setup/renku.md            RenkuLab guide (students + teaching team setup)
├── environment/              Dockerfile + requirements of the session image (Jupyter + IGM, main branch)
├── session1/                 1a/1b lectures (Marp, incl. Ebener case), 1c RenkuLab setup slides, 1d tutorial 1 (SIA toy model), 1e choose-your-glacier + candidate list
├── session2/                 lecture 2 (IGM, Marp), tutorial-2/ (IGM runs A, B, C)
├── session3/                 lecture 3 (project goals, report)
├── session4/mont-mine/       Mont Miné notebook (taught in the course 'Dates and Rates'), deglaciation vs sample ages
├── project/                  project guide + my-glacier/ template (YAML experiments, run_future.sh, analysis notebook)
├── utils/helper.py           plotting helpers used by all notebooks
└── archive/                  old scripts (not published)
```

- Slides: `marp session1/1a-lecture-glacier-processes.md` (Marp CLI or the VS Code Marp extension).
- Website (Jupyter Book): `./0-clean.sh; ./1-build-site.sh; ./3-publish-site.sh`.
- Course image: built by `.github/workflows/build-image.yml` and pushed to `ghcr.io/jouvetg/glamod:2026`.
