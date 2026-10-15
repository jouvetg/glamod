# glamod — Glacier Modelling course (UNIL, Master)

Course material for *Glacier Modelling* (G. Jouvet, UNIL): 3 afternoons × 4 hours, each one hour of lecture, about one hour of guided hands-on, then a mini-project on a Swiss glacier with [IGM 3.2](https://igm-model.org). Students run everything online on [RenkuLab](https://renkulab.io).

```
glamod/
├── infos.md                  course front page (programme, links)
├── setup/renku.md            RenkuLab guide (students + teaching team setup)
├── environment/              Dockerfile + requirements of the session image (Jupyter + IGM 3.2.1)
├── session1/                 lectures 1a/1b (Marp, incl. Ebener case), tutorial 1 (SIA toy model), choose-your-glacier + candidate list
├── session2/                 lecture 2 (IGM, Marp), glacier-goo break slides, tutorial-2/ (IGM runs A, B, C)
├── session3/                 lecture 3 (project goals, report)
├── session4/mont-mine/       Mont Miné notebook (taught in the course 'Dates and Rates'), deglaciation vs sample ages
├── project/                  project guide + my-glacier/ template (past / future experiments, notebook)
├── utils/helper.py           plotting helpers used by all notebooks
└── archive/                  old scripts (not published)
```

- Slides: `marp session1/lecture-1a-glacier-processes.md` (Marp CLI or the VS Code Marp extension).
- Website (Jupyter Book): `./0-clean.sh; ./1-build-site.sh; ./3-publish-site.sh`.
- Course image: built by `.github/workflows/build-image.yml` and pushed to `ghcr.io/jouvetg/glamod:2026`.
