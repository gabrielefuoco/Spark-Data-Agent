"""PySpark Medallion lakehouse package (Bronze, Silver, Gold, Manifest).

Nothing is re-exported here on purpose. Eager re-exports made `import
src.spark` fail whenever it ran first (`src.spark` -> `src.lakehouse.storage`
-> package `__init__` -> `bronze` -> `src.spark`, still half-initialised), a
cycle the test suite never hit because it happens to import the lakehouse
before Spark. Import from the concrete modules instead.
"""
