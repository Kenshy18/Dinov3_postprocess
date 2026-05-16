# EVA02 checkpoints

Place EVA02 runtime artifacts here:

```text
checkpoints/eva02/
  detector/model_final.pth
  classifier/best.pt
```

These paths are repo-local runtime artifacts. They may be hardlinked from the
validated training workspace during development, but normal clones restore them
from the shared Drive artifact folder.
