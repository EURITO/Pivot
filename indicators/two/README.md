Prerequisites
-------------

Please make sure that you have followed the instructions for getting started in the `README.md` at the base of this repository.

Topic modellig API description
------------------------------

Following the design pattern set out in `*_topics.py`, there must be three functions defined per dataset (`arxiv_topics`, `nih_topics`, `cordis_topics`):

- `get_lat_lon`: Which returns a list with items of the form `(institute_id, lat, lon)` for every institute in Europe in the dataset
- `get_iso2_to_id`: Which returns a list with items of the form `(object_id, iso2)` for every object (article or project) in the dataset (incl. non-European). `object_id` can clearly occur multiple times if there are multiple countries in the dataset.
- `get_objects`: Which returns every object in the dataset, in a general form of `list[dict]`, where each "row" is of the form `dict(id, text, title, created)`.

Adding a new module into `make_topics` after this is then trivial, assuming that a model configuration has also been added under `indicators/core/config/{dataset}.yaml`.

Step 1: topic modelling
-----------------------

Run the following to (re)generate the topics, which are save under a new `{dataset}-*` folder in this directory (note, this will not be versioned):

```base
python make_topics.py
```

