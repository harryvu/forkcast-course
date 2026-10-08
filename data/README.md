# Bundled data

Everything the tests and CI need lives here and stays small. Real datasets are downloaded in the Colab notebooks.

- `taxi_sample.csv` (Lesson 6): **simulated** trips with the exact column names of the NYC TLC yellow-taxi
  trip records (`tpep_pickup_datetime`, `tpep_dropoff_datetime`, `trip_distance`, `passenger_count`,
  `PULocationID`, `DOLocationID`), 6,000 rows, including deliberately impossible rows for the cleaning step.
  It exists so the Lesson 6 tests run offline; the lesson itself downloads a real month with `tools/get_taxi.py`.
- `reviews_mini.csv` (Lesson 11): 800 **templated** delivery reviews, half positive (label 1) and half negative
  (label 0), generated from a few sentence patterns so the Lesson 11 tests run offline. The lesson itself trains
  on real Yelp reviews streamed from the Hugging Face Hub.
