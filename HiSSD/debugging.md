
```bash
  File "miniconda3/envs/hissd/lib/python3.10/site-packages/pysc2/maps/lib.py", line 122, in get_maps
    raise DuplicateMapError("Duplicate map found: " + map_name)
pysc2.maps.lib.DuplicateMapError: Duplicate map found: 3m

```

To fix this, make comment in the function 
```python 
def get_maps():
  """Get the full dict of maps {map_name: map_class}."""
  maps = {}
  for mp in Map.all_subclasses():
    if mp.filename or mp.battle_net:
      map_name = mp.__name__
      # if map_name in maps:
      #   raise DuplicateMapError("Duplicate map found: " + map_name)
      maps[map_name] = mp
  return maps
```
