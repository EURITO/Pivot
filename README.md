Covid Pivot
===========

EURITO indicators for the covid pivot. See the `indicators` directory for more information.

Note: to use some of the packages under `indicators` you must first install `git-crypt` and then run `git-crypt unlock path/to/eurito.key`. To access `eurito.key`, please get in touch with one of the core developers.

Installation
============

From the base of the repo (i.e. `path/to/Pivot`)

```bash
export PYTHONPATH=$PWD:${PYTHONPATH}`
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

Finally you need to do ONE of the following:

* `pip install -r requirements.txt`

or

```
pip install PyYAML>=5.3.0
export PYTHONPATH=/path/to/nesta:$PWD:${PYTHONPATH}
```

The second option avoids the need to check out the enormous `nesta` repo, albeit the first option is more convenient.
