# Vision Transformer v1 — find the people in a photo

The notebook of the tutorial's vision track: a Vision Transformer, written from scratch, that labels every pixel of a
photo *person* or *background*. It learns from two group photos whose people were traced by hand, and it has a
labelling tool built in, so you can trace your own photos and train on them.

## What is in this folder

| | |
|---|---|
| `Vision_Transformer_V1.ipynb` | the notebook: 10 cells, from the settings to the attention map |
| `vision_people_data/` | the dataset — `people_1.jpg`, `people_2.jpg` and `outlines.json` (every person traced as polygons). The notebook reads this folder **next to it**, so keep the two together. |
| `requirements.txt` | the Python packages it needs |

## Run it on your own computer

You need **Python 3.9 or newer**. Check with `python3 --version` (Windows: `py --version`); if it is missing,
install it from [python.org](https://www.python.org/downloads/).

1. Open a terminal **in this `vision` folder**.
2. Make a virtual environment — a private set of packages for this notebook — and install them (a few minutes; PyTorch
   is the big one):

   macOS or Linux:

       python3 -m venv .venv
       source .venv/bin/activate
       pip install -r requirements.txt

   Windows (PowerShell):

       py -m venv .venv
       .venv\Scripts\Activate.ps1
       pip install -r requirements.txt

   With an NVIDIA graphics card, first install PyTorch with its CUDA support using the selector on
   [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.
3. Start Jupyter:

       jupyter lab

   Your browser opens JupyterLab. Double-click `Vision_Transformer_V1.ipynb` on the left.
4. **Run → Run All Cells.** Cell 1 prints the device it found: `cuda` (an NVIDIA GPU), `mps` (the GPU of an
   Apple-silicon Mac) or `cpu`.
5. Wait for cell 7, the training: 60 epochs. On an Apple-silicon Mac's GPU it took 11 minutes here
   (671 seconds); on a CPU it takes much longer. For a quick first try, set `EPOCHS = 5` in cell 1 (the
   scores will be lower).

Next time you only need steps 3 and 4 — after `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`) in
this folder.

## What you get

* **Cell 8** scores the model. In the tutorial's run it covers the people of the two photos it learned from with a
  person IoU of 93.2% and 92.8%. The same code on another computer can give slightly different numbers.
* **Cell 9** paints the people in any photo: put a photo in this folder and run
  `segment_photo("my_photo.jpg")`, or pick a colour: `segment_photo("my_photo.jpg", color="#00c853")`.
* **Cell 10** shows where one patch on a person looks (attention).
* The trained model is saved as `people_vit.pt` in this folder.

## The labelling tool (cell 3)

When cell 3 runs, an editor opens under it: the dataset as a tree on the left, the open photo with every person's
outline on the right.

* **Add photos…** copies your photos into `vision_people_data/` (a photo longer than 2400 pixels is shrunk first).
* **New person**: click corner points along a person's outline; double-click or press Enter to close it, Esc to cancel.
* Click a person to select them: round points appear, and a small square in the middle of every edge. Drag a point
  to move it; drag a square to pull a new point out of that edge (a click adds it right there). **More points** adds
  one to every edge at once, for a finer outline. Delete removes the selected point; Ctrl/Cmd + Z undoes.
* The mouse wheel zooms — zoomed in, shorter edges get squares too; dragging an empty spot moves the photo.
* **Add hole**: where an arm makes a loop (a hand on the hip), the wall seen through it is not the person. Select
  them, click Add hole and click around the gap; it is drawn dashed and left out of their mask. Deleting a hole's
  points down to two removes it.
* **Move behind / Move in front**: where two outlines overlap, the person in front keeps those pixels.
* The **train / test** mark on each photo: click it to switch.
* **Save** writes `vision_people_data/outlines.json`.

Then, to train on your labels: in cell 1 set `SPLIT = "labels"` (train on the photos marked *train*, test on those
marked *test*) and run all the cells again. Photos with nobody traced yet are left out. Keep a copy of
`outlines.json` before big changes — Save writes over it.

If the editor does not appear, `anywidget` is missing from the environment the notebook runs in: run
`pip install anywidget` there, restart the kernel (**Kernel → Restart Kernel**) and run the cells again.

## Other ways to run it

These two were not tried here; the notebook is written for them.

* **VS Code**: install the *Python* and *Jupyter* extensions, open this folder, open the notebook, choose the `.venv`
  environment as its kernel (*Select Kernel* at the top right), then *Run All*.
* **Google Colab**: upload the `vision_people_data` folder to the top of your Google Drive (*My Drive*), then open
  [colab.research.google.com](https://colab.research.google.com), *File → Upload notebook*. For faster training
  choose *Runtime → Change runtime type → T4 GPU*. *Runtime → Run all*; cell 1 asks for access to your Drive, where
  the dataset and your labels stay between sessions. Cell 3 installs `anywidget` itself. Colab forgets everything
  else when the session ends, including `people_vit.pt`.

## Tested with

Python 3.9, PyTorch 2.8.0, JupyterLab 4.5, anywidget 0.9 on macOS with an Apple-silicon GPU (`mps`).
Everything shown in the tutorial was measured with this notebook on that machine.

The two photos are the tutorial's example data: check their licence before you publish this folder.
